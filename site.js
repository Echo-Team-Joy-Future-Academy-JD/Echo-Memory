(function () {
  "use strict";

  const NAV_OFFSET = 72;
  const sectionEls = Array.from(document.querySelectorAll(".section[data-section]"));
  const navLinks = Array.from(document.querySelectorAll("[data-nav]"));
  const navToggle = document.getElementById("nav-toggle");
  const navMobile = document.getElementById("nav-menu-mobile");

  function setActiveNav(id) {
    navLinks.forEach(function (a) {
      a.classList.toggle("is-active", a.getAttribute("data-nav") === id);
    });
  }

  function scrollToSection(id) {
    var el = document.getElementById(id);
    if (!el) return;
    var top = el.getBoundingClientRect().top + window.scrollY - NAV_OFFSET;
    window.scrollTo({ top: top, behavior: "smooth" });
    setActiveNav(id);
    if (navMobile && !navMobile.hidden) {
      navToggle.setAttribute("aria-expanded", "false");
      navMobile.hidden = true;
      document.querySelector(".top-nav").classList.remove("is-open");
    }
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
      { root: null, rootMargin: "-" + NAV_OFFSET + "px 0px -55% 0px", threshold: 0 }
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

  if (navToggle && navMobile) {
    navToggle.addEventListener("click", function () {
      var open = navMobile.hidden;
      navMobile.hidden = !open;
      navToggle.setAttribute("aria-expanded", String(open));
      document.querySelector(".top-nav").classList.toggle("is-open", open);
    });
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
})();
