(function () {
  "use strict";

  const NAV_OFFSET = 72;
  const REDUCED_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const SCROLL_BEHAVIOR = REDUCED_MOTION ? "auto" : "smooth";

  const sectionEls = Array.from(document.querySelectorAll(".section[data-section]"));
  const navLinks = Array.from(document.querySelectorAll("[data-nav]"));
  const slideDots = Array.from(document.querySelectorAll(".slide-dots button[data-slide]"));
  const navToggle = document.getElementById("nav-toggle");
  const navMobile = document.getElementById("nav-menu-mobile");

  function setActiveNav(id) {
    navLinks.forEach(function (a) {
      a.classList.toggle("is-active", a.getAttribute("data-nav") === id);
    });
    slideDots.forEach(function (btn) {
      btn.classList.toggle("is-active", btn.getAttribute("data-slide") === id);
    });
  }

  function scrollToSection(id) {
    var el = document.getElementById(id);
    if (!el) return;
    el.scrollIntoView({ behavior: SCROLL_BEHAVIOR, block: "start" });
    setActiveNav(id);
    if (navMobile && !navMobile.hidden) {
      navToggle.setAttribute("aria-expanded", "false");
      navMobile.hidden = true;
      document.querySelector(".top-nav").classList.remove("is-open");
    }
  }

  function sectionIndex(id) {
    return sectionEls.findIndex(function (s) {
      return s.getAttribute("data-section") === id;
    });
  }

  function scrollAdjacent(delta) {
    var activeId = null;
    sectionEls.forEach(function (s) {
      var rect = s.getBoundingClientRect();
      if (rect.top <= NAV_OFFSET + 80 && rect.bottom > NAV_OFFSET + 80) {
        activeId = s.getAttribute("data-section");
      }
    });
    if (!activeId && sectionEls.length) activeId = sectionEls[0].getAttribute("data-section");
    var idx = sectionIndex(activeId);
    if (idx < 0) return;
    var next = sectionEls[idx + delta];
    if (next) scrollToSection(next.getAttribute("data-section"));
  }

  if (sectionEls.length) {
    var observer = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            setActiveNav(entry.target.getAttribute("data-section"));
          }
        });
      },
      { root: null, rootMargin: "-" + NAV_OFFSET + "px 0px -50% 0px", threshold: 0.12 }
    );
    sectionEls.forEach(function (section) {
      observer.observe(section);
    });
  }

  navLinks.forEach(function (a) {
    a.addEventListener("click", function (e) {
      e.preventDefault();
      scrollToSection(a.getAttribute("data-nav"));
    });
  });

  slideDots.forEach(function (btn) {
    btn.addEventListener("click", function () {
      scrollToSection(btn.getAttribute("data-slide"));
    });
  });

  if (navToggle && navMobile) {
    navToggle.addEventListener("click", function () {
      var open = navMobile.hidden;
      navMobile.hidden = !open;
      navToggle.setAttribute("aria-expanded", String(open));
      document.querySelector(".top-nav").classList.toggle("is-open", open);
    });
  }

  document.addEventListener("keydown", function (e) {
    if (e.target.closest("input, textarea, pre, [contenteditable]")) return;
    if (e.key === "ArrowDown" || e.key === "PageDown") {
      e.preventDefault();
      scrollAdjacent(1);
    } else if (e.key === "ArrowUp" || e.key === "PageUp") {
      e.preventDefault();
      scrollAdjacent(-1);
    }
  });

  function hydrateMetric(card) {
    var valueEl = card.querySelector("[data-metric-value]");
    var badgeUrl = card.getAttribute("data-badge-url");
    if (!valueEl || !badgeUrl) return;

    fetch(badgeUrl, { cache: "no-store" })
      .then(function (res) {
        if (!res.ok) throw new Error("Request failed");
        return res.json();
      })
      .then(function (data) {
        valueEl.textContent = data.value || data.message || valueEl.dataset.fallback || "—";
      })
      .catch(function () {
        valueEl.textContent = valueEl.dataset.fallback || "—";
      });
  }

  document.querySelectorAll("[data-badge-url]").forEach(function (card) {
    hydrateMetric(card);
  });

  document.querySelectorAll("[data-qual-viewer]").forEach(function (viewer) {
    var image = viewer.querySelector("[data-qual-image]");
    var caption = viewer.querySelector("[data-qual-caption]");
    var chips = viewer.querySelectorAll(".qual-chip");

    chips.forEach(function (chip) {
      chip.addEventListener("click", function () {
        chips.forEach(function (c) {
          c.classList.remove("is-active");
          c.setAttribute("aria-selected", "false");
        });
        chip.classList.add("is-active");
        chip.setAttribute("aria-selected", "true");
        if (image) {
          image.src = chip.dataset.src;
          image.alt = chip.dataset.alt || "";
        }
        if (caption) caption.textContent = chip.dataset.caption || "";
      });
    });
  });

  function initCiteSwitcher() {
    document.querySelectorAll("[data-cite-switcher]").forEach(function (switcher) {
      var chips = switcher.querySelectorAll(".cite-chip");
      var panels = switcher.querySelectorAll("[data-cite-panel]");

      chips.forEach(function (chip) {
        chip.addEventListener("click", function () {
          var key = chip.getAttribute("data-cite");
          chips.forEach(function (c) {
            c.classList.remove("is-active");
            c.setAttribute("aria-selected", "false");
          });
          chip.classList.add("is-active");
          chip.setAttribute("aria-selected", "true");
          panels.forEach(function (panel) {
            var match = panel.getAttribute("data-cite-panel") === key;
            panel.classList.toggle("is-active", match);
            panel.hidden = !match;
          });
        });
      });
    });
  }

  initCiteSwitcher();

  document.querySelectorAll("[data-copy-target]").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var targetId = btn.getAttribute("data-copy-target");
      var target = document.getElementById(targetId);
      if (!target) return;
      navigator.clipboard.writeText(target.textContent).then(
        function () {
          btn.textContent = "Copied";
          btn.classList.add("is-copied");
          setTimeout(function () {
            btn.textContent = "Copy";
            btn.classList.remove("is-copied");
          }, 2000);
        },
        function () {
          btn.textContent = "Failed";
        }
      );
    });
  });
})();
