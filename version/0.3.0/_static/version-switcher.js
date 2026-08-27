(() => {
  "use strict";

  const select = document.getElementById("blvpy-version-select");
  if (!select) {
    return;
  }

  const switcherUrl = select.dataset.switcherUrl;
  const currentVersion = select.dataset.currentVersion;
  if (!switcherUrl) {
    return;
  }

  fetch(switcherUrl, { credentials: "omit" })
    .then((response) => {
      if (!response.ok) {
        throw new Error(`version index returned ${response.status}`);
      }
      return response.json();
    })
    .then((entries) => {
      if (!Array.isArray(entries) || entries.length === 0) {
        return;
      }

      const currentNumberedEntry = entries.find(
        (entry) =>
          entry &&
          entry.preferred !== true &&
          entry.version === currentVersion &&
          typeof entry.url === "string",
      );
      let selectLatest = false;
      if (currentNumberedEntry) {
        const switcherLocation = new URL(switcherUrl, window.location.href);
        const numberedLocation = new URL(currentNumberedEntry.url, window.location.href);
        const numberedPath = numberedLocation.pathname.endsWith("/")
          ? numberedLocation.pathname
          : `${numberedLocation.pathname}/`;
        const onNumberedPath =
          window.location.pathname === numberedPath.slice(0, -1) ||
          window.location.pathname.startsWith(numberedPath);
        selectLatest = window.location.origin === switcherLocation.origin && !onNumberedPath;
      }

      select.replaceChildren();
      for (const entry of entries) {
        if (!entry || typeof entry.url !== "string") {
          continue;
        }
        const label = String(entry.name || entry.version || "unknown");
        const option = document.createElement("option");
        option.value = entry.url;
        option.textContent = label;
        option.selected = selectLatest
          ? entry.preferred === true
          : entry.preferred !== true && (entry.version === currentVersion || label === currentVersion);
        select.appendChild(option);
      }

      if (select.options.length === 0) {
        return;
      }
      select.hidden = false;
      const fallback = document.querySelector(".version-switcher-fallback");
      if (fallback) {
        fallback.hidden = true;
      }
    })
    .catch(() => {
      // A missing index must not interfere with reading a local or archived build.
    });

  select.addEventListener("change", () => {
    if (select.value) {
      window.location.assign(select.value);
    }
  });
})();
