"""Private human-readable progress reporting for BLVPY solves."""

from __future__ import annotations

import logging
import math
import re
import sys
import textwrap
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from time import perf_counter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .result import BilevelResult, IterationRecord, RunRecord

_WIDTH = 79
_PREFIX = "(BLVPY)"
_MAX_MESSAGE_LENGTH = 160
_HANDLER_MARKER = "_blvpy_progress_handler"
_ANSI_ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\)|[@-_])")
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f-\x9f]")


class _StderrHandler(logging.Handler):
    """A handler that follows the process's current stderr stream."""

    def emit(self, record: logging.LogRecord) -> None:
        message = self.format(record)
        print(message, file=sys.stderr)


def _install_handler() -> logging.Logger:
    """Configure the package logger once without touching the root logger."""

    logger = logging.getLogger("blvpy")
    if not any(getattr(handler, _HANDLER_MARKER, False) for handler in logger.handlers):
        handler = _StderrHandler()
        setattr(handler, _HANDLER_MARKER, True)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


_LOGGER = _install_handler()


def _package_version() -> str:
    try:
        return version("blvpy")
    except PackageNotFoundError:
        return "unknown"


@dataclass(slots=True)
class ProgressReporter:
    """Solve-scoped writer for the concise BLVPY progress transcript."""

    enabled: bool = False
    _phase_started_at: float = field(default_factory=perf_counter, init=False)
    _phase: str | None = field(default=None, init=False)
    _banner_emitted: bool = field(default=False, init=False)

    def problem(
        self,
        *,
        upper_dimension: int,
        lower_dimension: int,
        canonical_variables: int,
        canonical_constraints: int,
        zero: int,
        nonnegative: int,
        soc: tuple[int, ...],
        lower_solver: str,
        nonlinear_solver: str,
        best_of: int | None,
        requested_runs: int,
        epsilon_initial: float,
        epsilon_target: float,
        contraction: float,
    ) -> None:
        """Write validated problem dimensions and numerical configuration."""

        try:
            if not self.enabled:
                return
            self._banner()
            self._section("Problem")
            self._detail(
                "Dimensions",
                f"upper={upper_dimension}",
                f"lower={lower_dimension}",
                f"canonical_variables={canonical_variables}",
                f"canonical_constraints={canonical_constraints}",
            )
            soc_text = "[" + ", ".join(str(dimension) for dimension in soc) + "]"
            self._detail("Cones", f"zero={zero}", f"nonnegative={nonnegative}", f"soc={soc_text}")
            self._detail(
                "Solvers",
                f"lower={_clean_text(lower_solver)}",
                f"nonlinear={_clean_text(nonlinear_solver)}",
            )
            if best_of is None:
                self._detail("Search", "mode=deterministic", f"runs={requested_runs}")
            else:
                self._detail("Search", "mode=random", f"best_of={best_of}")
            self._detail(
                "Epsilon",
                f"initial={_number(epsilon_initial)}",
                f"target={_number(epsilon_target)}",
                f"contraction={_number(contraction)}",
            )
        except Exception:
            return

    def initialization(self) -> None:
        """Open initialization reporting before run preparation begins."""

        try:
            if not self.enabled:
                return
            self._section("Initialization")
        except Exception:
            return

    def run_begin(
        self,
        *,
        run_index: int,
        total_runs: int,
        randomized: bool,
    ) -> None:
        """Report the beginning of one complete solution run."""

        try:
            if not self.enabled:
                return
            mode = "random" if randomized else "deterministic"
            self._line(f"Run {run_index + 1}/{total_runs}", "begin", f"mode={mode}")
        except Exception:
            return

    def projection(
        self,
        *,
        run_index: int,
        total_runs: int,
        before_violation: float | None = None,
    ) -> None:
        """Report an upper-point projection when that path is executed."""

        try:
            if not self.enabled:
                return
            fields = [f"run={run_index + 1}/{total_runs}"]
            if before_violation is not None:
                fields.append(f"before_violation={_number(before_violation)}")
            self._line("Projection", *fields)
        except Exception:
            return

    def restoration(
        self,
        *,
        run_index: int,
        total_runs: int,
        before_violation: float | None = None,
    ) -> None:
        """Report a lifted feasibility-restoration attempt."""

        try:
            if not self.enabled:
                return
            fields = [f"run={run_index + 1}/{total_runs}"]
            if before_violation is not None:
                fields.append(f"before_violation={_number(before_violation)}")
            self._line("Restoration", *fields)
        except Exception:
            return

    def run(
        self,
        *,
        run_index: int,
        total_runs: int,
        record: RunRecord,
    ) -> None:
        """Write the terminal outcome of one complete solution run."""

        try:
            if not self.enabled:
                return
            decision = "succeeded" if record.succeeded else "failed"
            self._detail(
                f"Run {run_index + 1}/{total_runs}",
                decision,
                f"status={_clean_text(record.status)}",
            )
            final = record.final_iteration
            self._record_details(
                objective=record.objective,
                residuals=record.residuals,
                solve_time=None if final is None else final.solve_time,
                num_iters=None if final is None else final.num_iters,
            )
            if not record.succeeded and record.message:
                self._indented(f"message={_clean_text(record.message)}")
        except Exception:
            return

    def selected_run(
        self,
        *,
        run_index: int,
        total_runs: int,
        objective: float | None,
    ) -> None:
        """Report the completed run selected as the returned result."""

        try:
            if not self.enabled:
                return
            self._line(
                f"Selected run {run_index + 1}/{total_runs}",
                *_objective_fields(objective),
            )
        except Exception:
            return

    def continuation(self) -> None:
        """Open the continuation section."""

        try:
            if not self.enabled:
                return
            self._section("Continuation")
        except Exception:
            return

    def attempt(
        self,
        *,
        run_index: int,
        total_runs: int,
        attempt_index: int,
        kind: str,
        epsilon: float,
        record: IterationRecord,
        accepted: bool,
    ) -> None:
        """Write one scheduled or inserted continuation solve outcome."""

        try:
            if not self.enabled:
                return
            kind_text = _clean_text(kind)
            decision = "accepted" if accepted else "rejected"
            self._detail(
                f"Run {run_index + 1}/{total_runs}, attempt {attempt_index + 1} [{kind_text}]",
                decision,
                f"eps={_number(epsilon)}",
                f"status={_clean_text(record.status)}",
            )
            self._record_details(
                objective=record.objective,
                residuals=record.residuals,
                solve_time=record.solve_time,
                num_iters=record.num_iters,
            )
            if not accepted and record.message:
                self._indented(f"message={_clean_text(record.message)}")
        except Exception:
            return

    def inserting_epsilon(
        self,
        *,
        run_index: int,
        total_runs: int,
        epsilon: float,
        target: float,
    ) -> None:
        """Report insertion of an intermediate continuation tolerance."""

        try:
            if not self.enabled:
                return
            self._line(
                "Inserting epsilon",
                f"run={run_index + 1}/{total_runs}",
                f"epsilon={_number(epsilon)}",
                f"target={_number(target)}",
            )
        except Exception:
            return

    def retry_exhausted(
        self,
        *,
        run_index: int,
        total_runs: int,
        last_successful_epsilon: float,
        max_retries: int,
    ) -> None:
        """Report exhaustion of the continuation retry budget."""

        try:
            if not self.enabled:
                return
            self._line(
                "Retry budget exhausted",
                f"run={run_index + 1}/{total_runs}",
                f"retries={max_retries}",
                f"last_successful_epsilon={_number(last_successful_epsilon)}",
            )
        except Exception:
            return

    def summary(
        self,
        *,
        result: BilevelResult,
        successful_runs: int,
        requested_runs: int,
        accepted_attempts: int,
        attempted_solves: int,
        elapsed: float | None,
    ) -> None:
        """Write the terminal result and local-solution qualification."""

        try:
            if not self.enabled:
                return
            self._section("Summary")
            status_fields = [f"status={_clean_text(result.status)}"]
            status_fields.extend(_objective_fields(result.objective))
            if result.final_epsilon is not None:
                status_fields.append(f"final_epsilon={_number(result.final_epsilon)}")
            if elapsed is not None:
                status_fields.append(f"elapsed={_number(elapsed)}s")
            self._detail("Status", *status_fields)
            if result.residuals is not None:
                self._detail("Residuals", *_residual_fields(result.residuals))
            self._detail(
                "Progress",
                f"accepted={accepted_attempts}",
                f"attempted={attempted_solves}",
                f"successful_runs={successful_runs}/{requested_runs}",
            )
            if result.message:
                self._detail("Message", _clean_text(result.message))
            self._write(f"{_PREFIX} Local numerical solution; not a rigorous bilevel certificate.")
        except Exception:
            return

    def failure(self, error: BaseException, *, elapsed: float | None) -> None:
        """Write a failed terminal summary without changing the exception."""

        try:
            if not self.enabled:
                return
            self._section("Summary")
            fields = ["status=failed"]
            if elapsed is not None:
                fields.append(f"elapsed={_number(elapsed)}s")
            fields.append(f"error={_clean_text(f'{type(error).__name__}: {error}')}")
            self._detail("Status", *fields)
        except Exception:
            return

    def _section(self, title: str) -> None:
        self._banner()
        now = perf_counter()
        if self._phase is not None:
            self._line(f"{self._phase} complete", f"elapsed={_number(now - self._phase_started_at)}s")
        self._write("-" * _WIDTH)
        self._write(title.center(_WIDTH))
        self._write("-" * _WIDTH)
        self._phase = title
        self._phase_started_at = now

    def _banner(self) -> None:
        if self._banner_emitted:
            return
        self._write("=" * _WIDTH)
        self._write("BLVPY".center(_WIDTH))
        self._write(f"v{_package_version()}".center(_WIDTH))
        self._write("=" * _WIDTH)
        self._banner_emitted = True

    def _line(self, label: str, *fields: str) -> None:
        self._write_fields(f"{_PREFIX} {label}", *fields, first_separator=" | ")

    def _detail(self, label: str, *fields: str) -> None:
        self._write_fields(f"{_PREFIX} {label}:", *fields)

    def _indented(self, *fields: str) -> None:
        if fields:
            self._write_fields(f"{_PREFIX}  ", *fields)

    def _write_fields(
        self,
        leader: str,
        *fields: str,
        first_separator: str = " ",
    ) -> None:
        remaining = [field for field in fields if field]
        if not remaining:
            self._write(leader)
            return
        current = leader
        for value in remaining:
            separator = first_separator if current == leader else " | "
            candidate = f"{current}{separator}{value}"
            if len(candidate) <= _WIDTH or current == leader:
                current = candidate
                continue
            self._write(current)
            current = f"{_PREFIX}   {value}"
        self._write(current)

    def _record_details(
        self,
        *,
        objective: float | None,
        residuals: Any | None,
        solve_time: float | None,
        num_iters: int | None,
    ) -> None:
        numerical = _objective_fields(objective)
        diagnostics: list[str] = []
        if residuals is not None:
            numerical.extend(
                [
                    f"feasibility={_number(residuals.max_feasibility)}",
                    f"gap={_number(residuals.gap_violation)}",
                ]
            )
            diagnostics.append(f"complementarity={_number(residuals.complementarity)}")
        if solve_time is not None:
            diagnostics.append(f"time={_number(solve_time)}s")
        if num_iters is not None:
            diagnostics.append(f"iters={num_iters}")
        self._indented(*numerical)
        self._indented(*diagnostics)

    def _write(self, message: str) -> None:
        if len(message) <= _WIDTH:
            _LOGGER.info(message)
            return
        subsequent_indent = f"{_PREFIX}   " if message.startswith(_PREFIX) else ""
        lines = textwrap.wrap(
            message,
            width=_WIDTH,
            subsequent_indent=subsequent_indent,
            break_long_words=True,
            break_on_hyphens=False,
        )
        for line in lines or [""]:
            _LOGGER.info(line)


def _objective_fields(objective: float | None) -> list[str]:
    return [] if objective is None else [f"objective={_number(objective)}"]


def _residual_fields(residuals: Any | None) -> list[str]:
    if residuals is None:
        return []
    return [
        f"max_feasibility={_number(residuals.max_feasibility)}",
        f"gap_violation={_number(residuals.gap_violation)}",
        f"complementarity={_number(residuals.complementarity)}",
    ]


def _number(value: Any) -> str:
    if value is None:
        return "none"
    number = float(value)
    if math.isnan(number):
        return "nan"
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    return f"{number:.3e}"


def _clean_text(value: Any) -> str:
    text = _ANSI_ESCAPE.sub("", str(value))
    text = _CONTROL_CHARACTERS.sub(" ", text)
    text = " ".join(text.split()) or "none"
    if len(text) <= _MAX_MESSAGE_LENGTH:
        return text
    return text[: _MAX_MESSAGE_LENGTH - 3].rstrip() + "..."
