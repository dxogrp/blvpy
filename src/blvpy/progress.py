"""Private human-readable progress reporting for BLVPY solves."""

from __future__ import annotations

import logging
import math
import re
import sys
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from time import perf_counter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .result import BilevelResult, IterationRecord, StartRecord

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
        requested_starts: int,
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
                f"starts={requested_starts}",
            )
            self._detail(
                "Epsilon",
                f"initial={_number(epsilon_initial)}",
                f"target={_number(epsilon_target)}",
                f"contraction={_number(contraction)}",
            )
        except Exception:
            return

    def initialization(self) -> None:
        """Open initialization reporting before start preparation begins."""

        try:
            if not self.enabled:
                return
            self._section("Initialization")
        except Exception:
            return

    def starts(self, *, requested_starts: int, deduplicated_starts: int) -> None:
        """Describe the requested and distinct initialization starts."""

        try:
            if not self.enabled:
                return
            self._detail("Starts", f"requested={requested_starts}", f"unique={deduplicated_starts}")
        except Exception:
            return

    def projection(
        self,
        *,
        start_index: int,
        total_starts: int,
        before_violation: float | None = None,
    ) -> None:
        """Report an upper-start projection when that path is executed."""

        try:
            if not self.enabled:
                return
            fields = [f"start={start_index + 1}/{total_starts}"]
            if before_violation is not None:
                fields.append(f"before_violation={_number(before_violation)}")
            self._line("Projection", *fields)
        except Exception:
            return

    def restoration(
        self,
        *,
        start_index: int,
        total_starts: int,
        before_violation: float | None = None,
    ) -> None:
        """Report a lifted feasibility-restoration attempt."""

        try:
            if not self.enabled:
                return
            fields = [f"start={start_index + 1}/{total_starts}"]
            if before_violation is not None:
                fields.append(f"before_violation={_number(before_violation)}")
            self._line("Restoration", *fields)
        except Exception:
            return

    def start(
        self,
        *,
        start_index: int,
        total_starts: int,
        record: StartRecord,
        accepted: bool,
        solve_time: float | None = None,
        num_iters: int | None = None,
    ) -> None:
        """Write the terminal outcome of one initialization attempt."""

        try:
            if not self.enabled:
                return
            fields = ["accepted" if accepted else "rejected", f"status={_clean_text(record.status)}"]
            fields.extend(_objective_fields(record.objective))
            fields.extend(_residual_fields(record.residuals))
            if solve_time is not None:
                fields.append(f"solve_time={_number(solve_time)}s")
            if num_iters is not None:
                fields.append(f"iterations={num_iters}")
            if not accepted and record.message:
                fields.append(f"message={_clean_text(record.message)}")
            self._line(f"Start {start_index + 1}/{total_starts}", *fields)
        except Exception:
            return

    def selected_start(
        self,
        *,
        start_index: int,
        total_starts: int,
        objective: float | None,
    ) -> None:
        """Report the restored start selected for continuation."""

        try:
            if not self.enabled:
                return
            self._line(
                f"Selected start {start_index + 1}/{total_starts}",
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
        attempt_index: int,
        kind: str,
        epsilon: float,
        record: IterationRecord,
        accepted: bool,
        start_index: int | None = None,
    ) -> None:
        """Write one scheduled, alternative-start, or inserted solve outcome."""

        try:
            if not self.enabled:
                return
            fields = [_clean_text(kind), f"epsilon={_number(epsilon)}"]
            if start_index is not None:
                fields.append(f"start={start_index + 1}")
            fields.extend(["accepted" if accepted else "rejected", f"status={_clean_text(record.status)}"])
            fields.extend(_objective_fields(record.objective))
            fields.extend(_residual_fields(record.residuals))
            if record.solve_time is not None:
                fields.append(f"solve_time={_number(record.solve_time)}s")
            if record.num_iters is not None:
                fields.append(f"iterations={record.num_iters}")
            if not accepted and record.message:
                fields.append(f"message={_clean_text(record.message)}")
            self._line(f"Attempt {attempt_index + 1}", *fields)
        except Exception:
            return

    def inserting_epsilon(self, *, epsilon: float, target: float) -> None:
        """Report insertion of an intermediate continuation tolerance."""

        try:
            if not self.enabled:
                return
            self._write(f"{_PREFIX} Inserting epsilon={_number(epsilon)} before retrying target={_number(target)}")
        except Exception:
            return

    def retry_exhausted(self, *, last_successful_epsilon: float, max_retries: int) -> None:
        """Report exhaustion of the continuation retry budget."""

        try:
            if not self.enabled:
                return
            self._line(
                "Retry budget exhausted",
                f"retries={max_retries}",
                f"last_successful_epsilon={_number(last_successful_epsilon)}",
            )
        except Exception:
            return

    def summary(
        self,
        *,
        result: BilevelResult,
        successful_starts: int,
        requested_starts: int,
        accepted_attempts: int,
        attempted_solves: int,
        elapsed: float | None,
    ) -> None:
        """Write the terminal result and local-solution qualification."""

        try:
            if not self.enabled:
                return
            self._section("Summary")
            status_fields = [_clean_text(result.status)]
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
                f"successful_starts={successful_starts}/{requested_starts}",
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
            fields = ["failed"]
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
        suffix = " | ".join(field for field in fields if field)
        self._write(f"{_PREFIX} {label}" + (f" | {suffix}" if suffix else ""))

    def _detail(self, label: str, *fields: str) -> None:
        suffix = " | ".join(field for field in fields if field)
        self._write(f"{_PREFIX} {label}:" + (f" {suffix}" if suffix else ""))

    def _write(self, message: str) -> None:
        _LOGGER.info(message)


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
