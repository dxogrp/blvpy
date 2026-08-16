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

      select.replaceChildren();
      for (const entry of entries) {
        if (!entry || typeof entry.url !== "string") {
          continue;
        }
        const label = String(entry.name || entry.version || "unknown");
        const option = document.createElement("option");
        option.value = entry.url;
        option.textContent = entry.preferred ? `${label} (latest)` : label;
        option.selected = entry.version === currentVersion || label === currentVersion;
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

