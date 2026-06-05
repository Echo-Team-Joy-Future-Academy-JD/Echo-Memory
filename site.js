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
  const topNav = document.querySelector(".top-nav");
  const progressBar = document.querySelector(".scroll-progress__bar");
  const backTop = document.getElementById("back-top");
  const lightbox = document.getElementById("lightbox");
  const lightboxImg = document.getElementById("lightbox-img");
  const lightboxCaption = document.getElementById("lightbox-caption");
  const lightboxClose = lightbox ? lightbox.querySelector(".lightbox__close") : null;

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
      if (topNav) topNav.classList.remove("is-open");
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

  document.querySelectorAll(".reveal").forEach(function (el) {
    var revealObserver = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-visible");
            revealObserver.unobserve(entry.target);
          }
        });
      },
      { root: null, rootMargin: "0px 0px -8% 0px", threshold: 0.08 }
    );
    if (REDUCED_MOTION) {
      el.classList.add("is-visible");
    } else {
      revealObserver.observe(el);
    }
  });

  function updateScrollUI() {
    var scrollTop = window.scrollY || document.documentElement.scrollTop;
    var docHeight = document.documentElement.scrollHeight - window.innerHeight;
    var progress = docHeight > 0 ? (scrollTop / docHeight) * 100 : 0;

    if (progressBar) progressBar.style.width = progress + "%";
    if (topNav) topNav.classList.toggle("is-scrolled", scrollTop > 24);
    if (backTop) {
      var show = scrollTop > window.innerHeight * 0.6;
      backTop.hidden = !show;
      backTop.classList.toggle("is-visible", show);
    }
  }

  window.addEventListener("scroll", updateScrollUI, { passive: true });
  updateScrollUI();

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
      if (topNav) topNav.classList.toggle("is-open", open);
    });
  }

  if (backTop) {
    backTop.addEventListener("click", function () {
      scrollToSection("hero");
    });
  }

  function openLightbox(img, captionText) {
    if (!lightbox || !lightboxImg) return;
    lightboxImg.src = img.src;
    lightboxImg.alt = img.alt || "";
    if (lightboxCaption) {
      lightboxCaption.textContent = captionText || "";
    }
    lightbox.hidden = false;
    document.body.style.overflow = "hidden";
    if (lightboxClose) lightboxClose.focus();
  }

  function closeLightbox() {
    if (!lightbox) return;
    lightbox.hidden = true;
    document.body.style.overflow = "";
    if (lightboxImg) lightboxImg.src = "";
  }

  document.querySelectorAll("[data-zoomable]").forEach(function (figure) {
    figure.addEventListener("click", function (e) {
      if (e.target.closest(".qual-chip, .copy-btn, button, a")) return;
      var img = figure.querySelector("img.is-active, img[data-qual-image], img");
      if (!img || !img.src) return;
      var cap = figure.querySelector("figcaption");
      var capText = cap ? cap.textContent.replace(/Click to expand/g, "").trim() : "";
      openLightbox(img, capText);
    });
  });

  if (lightboxClose) {
    lightboxClose.addEventListener("click", closeLightbox);
  }

  if (lightbox) {
    lightbox.addEventListener("click", function (e) {
      if (e.target === lightbox) closeLightbox();
    });
  }

  document.addEventListener("keydown", function (e) {
    if (e.target.closest("input, textarea, pre, [contenteditable]")) return;

    if (e.key === "Escape" && lightbox && !lightbox.hidden) {
      closeLightbox();
      return;
    }

    if (e.key === "ArrowDown" || e.key === "PageDown") {
      e.preventDefault();
      scrollAdjacent(1);
    } else if (e.key === "ArrowUp" || e.key === "PageUp") {
      e.preventDefault();
      scrollAdjacent(-1);
    }
  });

  function animateMetricValue(el, target) {
    if (REDUCED_MOTION || !target || target === "—") {
      el.textContent = target;
      return;
    }
    var numeric = parseInt(String(target).replace(/,/g, ""), 10);
    if (isNaN(numeric)) {
      el.textContent = target;
      return;
    }
    var start = 0;
    var duration = 700;
    var startTime = null;
    function step(ts) {
      if (!startTime) startTime = ts;
      var t = Math.min((ts - startTime) / duration, 1);
      var eased = 1 - Math.pow(1 - t, 3);
      el.textContent = Math.round(start + (numeric - start) * eased).toLocaleString();
      if (t < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

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
        var val = data.value || data.message || valueEl.dataset.fallback || "—";
        animateMetricValue(valueEl, val);
      })
      .catch(function () {
        valueEl.textContent = valueEl.dataset.fallback || "—";
      });
  }

  document.querySelectorAll("[data-badge-url]").forEach(function (card) {
    var metricObserver = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            hydrateMetric(entry.target);
            metricObserver.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.4 }
    );
    metricObserver.observe(card);
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
        if (!image) return;

        var nextSrc = chip.dataset.src;
        var nextAlt = chip.dataset.alt || "";
        var nextCaption = chip.dataset.caption || "";

        var currentPath = new URL(image.src, window.location.href).pathname;
        if (currentPath.endsWith(nextSrc)) return;

        image.classList.add("is-fading");
        var preload = new Image();
        preload.onload = function () {
          image.src = nextSrc;
          image.alt = nextAlt;
          image.classList.remove("is-fading");
          image.classList.add("is-active");
        };
        preload.onerror = function () {
          image.classList.remove("is-fading");
        };
        preload.src = nextSrc;

        if (caption) {
          caption.innerHTML =
            nextCaption + ' <span class="zoom-hint">Click to expand</span>';
        }
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
