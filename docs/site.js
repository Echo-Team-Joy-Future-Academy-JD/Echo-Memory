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

  const overviewCaptions = {
    context:
      "<strong>Context memory</strong> — raw history windows (K = 1 / 5 / 20) carried across chunk-wise generation.",
    compression:
      "<strong>Compression memory</strong> — compact tokens at ratio r = 4 trade capacity for efficiency.",
    spatial:
      "<strong>Spatial memory</strong> — explicit read/write state targets layout and viewpoint carry.",
    ssm:
      "<strong>State-space memory</strong> — block-wise SSM updates stabilize long-horizon return.",
  };

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
    var navObserver = new IntersectionObserver(
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
      navObserver.observe(section);
    });

    var revealObserver = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-visible");
          }
        });
      },
      { root: null, threshold: 0.08 }
    );
    sectionEls.forEach(function (section) {
      revealObserver.observe(section);
      if (section.getBoundingClientRect().top < window.innerHeight * 0.9) {
        section.classList.add("is-visible");
      }
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

  function parseMetricNumber(text) {
    var raw = String(text || "").trim();
    if (!raw || raw === "—") return null;
    var m = raw.match(/^([\d.]+)\s*([kKmM])?/);
    if (!m) return null;
    var n = parseFloat(m[1]);
    if (m[2]) {
      var suffix = m[2].toLowerCase();
      if (suffix === "k") n *= 1000;
      if (suffix === "m") n *= 1000000;
    }
    return Math.round(n);
  }

  function formatMetricDisplay(n) {
    if (n >= 1000) return (n / 1000).toFixed(1).replace(/\.0$/, "") + "k";
    return String(n);
  }

  function animateMetricValue(el, targetText) {
    var end = parseMetricNumber(targetText);
    if (end === null || REDUCED_MOTION) {
      el.textContent = targetText;
      return;
    }
    var start = 0;
    var duration = 500;
    var startTime = null;
    function frame(ts) {
      if (!startTime) startTime = ts;
      var t = Math.min(1, (ts - startTime) / duration);
      var eased = 1 - Math.pow(1 - t, 3);
      var current = Math.round(start + (end - start) * eased);
      el.textContent = formatMetricDisplay(current);
      if (t < 1) requestAnimationFrame(frame);
      else el.textContent = targetText;
    }
    requestAnimationFrame(frame);
  }

  function hydrateMetric(card) {
    var valueEl = card.querySelector("[data-metric-value]");
    var badgeUrl = card.getAttribute("data-badge-url");
    if (!valueEl || !badgeUrl) return;

    card.classList.add("is-loading");
    fetch(badgeUrl, { cache: "no-store" })
      .then(function (res) {
        if (!res.ok) throw new Error("Request failed");
        return res.json();
      })
      .then(function (data) {
        var text = data.value || data.message || valueEl.dataset.fallback || "—";
        card.classList.remove("is-loading");
        animateMetricValue(valueEl, text);
      })
      .catch(function () {
        card.classList.remove("is-loading");
        valueEl.textContent = valueEl.dataset.fallback || "—";
      });
  }

  document.querySelectorAll("[data-badge-url]").forEach(function (card) {
    hydrateMetric(card);
  });

  function initOverviewDiagram() {
    var root = document.getElementById("diagram-overview");
    if (!root) return;
    var caption = root.querySelector("[data-overview-caption]");
    var memNodes = root.querySelectorAll("[data-memory]");

    function selectMemory(key) {
      root.classList.add("is-ready");
      memNodes.forEach(function (node) {
        node.classList.toggle("is-active", node.getAttribute("data-memory") === key);
      });
      if (caption && overviewCaptions[key]) {
        caption.innerHTML = overviewCaptions[key];
      }
    }

    memNodes.forEach(function (node) {
      var key = node.getAttribute("data-memory");
      node.addEventListener("click", function () {
        selectMemory(key);
      });
      node.addEventListener("keydown", function (e) {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          selectMemory(key);
        }
      });
    });
    selectMemory("context");
  }

  function initDesignDiagram() {
    var root = document.getElementById("diagram-design");
    var cardsWrap = document.querySelector("[data-design-cards]");
    if (!root) return;

    var cells = root.querySelectorAll("[data-design-cell]");
    var spokes = root.querySelectorAll("[data-spoke]");
    var cards = cardsWrap ? cardsWrap.querySelectorAll("[data-design]") : [];
    var caption = root.querySelector("[data-design-caption]");

    function selectDesign(key) {
      if (cardsWrap) cardsWrap.classList.add("has-selection");
      cells.forEach(function (c) {
        c.classList.toggle("is-active", c.getAttribute("data-design-cell") === key);
      });
      spokes.forEach(function (s) {
        s.classList.toggle("is-active", s.getAttribute("data-spoke") === key);
      });
      cards.forEach(function (c) {
        c.classList.toggle("is-active", c.getAttribute("data-design") === key);
      });
      if (caption) {
        var titles = {
          context: "Context — history windows K ∈ {1, 5, 20}.",
          compression: "Compression — learned tokens at ratio r = 4.",
          spatial: "Spatial — explicit read/write spatial state.",
          ssm: "State-Space — block-wise SSM for long-horizon carry.",
        };
        caption.textContent = titles[key] || caption.textContent;
      }
    }

    function bind(el, key) {
      el.addEventListener("click", function () {
        selectDesign(key);
      });
      el.addEventListener("keydown", function (e) {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          selectDesign(key);
        }
      });
    }

    cells.forEach(function (c) {
      bind(c, c.getAttribute("data-design-cell"));
    });
    cards.forEach(function (c) {
      bind(c, c.getAttribute("data-design"));
    });
    selectDesign("context");
  }

  function initQualViewer() {
    document.querySelectorAll("[data-qual-viewer]").forEach(function (viewer) {
      var images = viewer.querySelectorAll("[data-qual-image]");
      var caption = viewer.querySelector("[data-qual-caption]");
      var chips = viewer.querySelectorAll(".qual-chip");

      chips.forEach(function (chip) {
        chip.addEventListener("click", function () {
          var idx = chip.getAttribute("data-qual-index");
          chips.forEach(function (c) {
            c.classList.remove("is-active");
            c.setAttribute("aria-selected", "false");
          });
          chip.classList.add("is-active");
          chip.setAttribute("aria-selected", "true");
          images.forEach(function (img) {
            var match = img.getAttribute("data-qual-index") === idx;
            img.classList.toggle("is-visible", match);
          });
          if (caption) caption.textContent = chip.dataset.caption || "";
        });
      });
    });
  }

  var copyBtn = document.getElementById("copy-bibtex");
  var bibtexBlock = document.getElementById("bibtex-block");
  if (copyBtn && bibtexBlock) {
    copyBtn.addEventListener("click", function () {
      navigator.clipboard.writeText(bibtexBlock.textContent).then(
        function () {
          copyBtn.textContent = "Copied";
          copyBtn.classList.add("is-copied");
          setTimeout(function () {
            copyBtn.textContent = "Copy";
            copyBtn.classList.remove("is-copied");
          }, 2000);
        },
        function () {
          copyBtn.textContent = "Failed";
        }
      );
    });
  }

  initOverviewDiagram();
  initDesignDiagram();
  initQualViewer();
})();
