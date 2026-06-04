(function () {
  "use strict";

  const REPO = "Echo-Team-Joy-Future-Academy-JD/Echo-Memory";
  const REPO_API = "https://api.github.com/repos/" + REPO;

  const slidesRoot = document.getElementById("slides");
  const slideEls = Array.from(document.querySelectorAll(".slide[data-slide]"));
  const navLinks = Array.from(document.querySelectorAll("[data-nav]"));
  const dotBtns = Array.from(document.querySelectorAll(".slide-dots button[data-slide]"));
  const navToggle = document.getElementById("nav-toggle");
  const navMobile = document.getElementById("nav-menu-mobile");

  function formatCount(n) {
    if (typeof n !== "number" || Number.isNaN(n)) return "—";
    if (n >= 1000) return (n / 1000).toFixed(1).replace(/\.0$/, "") + "k";
    return String(n);
  }

  function setStat(name, value) {
    document.querySelectorAll('[data-stat="' + name + '"]').forEach(function (el) {
      el.textContent = value;
    });
  }

  function loadGitHubStats() {
    fetch(REPO_API, {
      headers: { Accept: "application/vnd.github+json" },
    })
      .then(function (res) {
        if (!res.ok) throw new Error("GitHub API " + res.status);
        return res.json();
      })
      .then(function (data) {
        var stars = formatCount(data.stargazers_count);
        var forks = formatCount(data.forks_count);
        var issues = formatCount(data.open_issues_count);
        setStat("stars", stars);
        setStat("forks", forks);
        setStat("issues", issues);
        var navCount = document.getElementById("nav-star-count");
        if (navCount) navCount.textContent = stars;
      })
      .catch(function () {
        setStat("stars", "—");
        setStat("forks", "—");
        setStat("issues", "—");
      });
  }

  function setActiveSlide(id) {
    navLinks.forEach(function (a) {
      a.classList.toggle("is-active", a.getAttribute("data-nav") === id);
    });
    dotBtns.forEach(function (btn) {
      btn.classList.toggle("is-active", btn.getAttribute("data-slide") === id);
    });
  }

  function scrollToSlide(id) {
    var el = document.getElementById(id);
    if (!el || !slidesRoot) return;
    el.scrollIntoView({ behavior: "smooth", block: "start" });
    setActiveSlide(id);
    if (navMobile && !navMobile.hidden) {
      navToggle.setAttribute("aria-expanded", "false");
      navMobile.hidden = true;
      document.querySelector(".top-nav").classList.remove("is-open");
    }
  }

  function currentSlideIndex() {
    var rootTop = slidesRoot.getBoundingClientRect().top;
    var marker = rootTop + slidesRoot.clientHeight * 0.35;
    var idx = 0;
    slideEls.forEach(function (slide, i) {
      var top = slide.getBoundingClientRect().top;
      if (top <= marker) idx = i;
    });
    return idx;
  }

  if (slidesRoot && slideEls.length) {
    var observer = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting && entry.intersectionRatio >= 0.45) {
            setActiveSlide(entry.target.getAttribute("data-slide"));
          }
        });
      },
      { root: slidesRoot, threshold: [0.45, 0.6] }
    );
    slideEls.forEach(function (slide) {
      observer.observe(slide);
    });

    slidesRoot.addEventListener(
      "scroll",
      function () {
        window.requestAnimationFrame(function () {
          var idx = currentSlideIndex();
          var id = slideEls[idx] && slideEls[idx].getAttribute("data-slide");
          if (id) setActiveSlide(id);
        });
      },
      { passive: true }
    );
  }

  navLinks.forEach(function (a) {
    a.addEventListener("click", function (e) {
      e.preventDefault();
      scrollToSlide(a.getAttribute("data-nav"));
    });
  });

  if (navMobile) {
    Array.from(navMobile.querySelectorAll("a[data-nav]")).forEach(function (a) {
      a.addEventListener("click", function (e) {
        e.preventDefault();
        scrollToSlide(a.getAttribute("data-nav"));
      });
    });
  }

  dotBtns.forEach(function (btn) {
    btn.addEventListener("click", function () {
      scrollToSlide(btn.getAttribute("data-slide"));
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
    if (!slidesRoot || e.target.closest("pre, input, textarea")) return;
    var keys = ["ArrowDown", "ArrowUp", "PageDown", "PageUp"];
    if (keys.indexOf(e.key) === -1) return;
    e.preventDefault();
    var idx = currentSlideIndex();
    if (e.key === "ArrowDown" || e.key === "PageDown") idx = Math.min(idx + 1, slideEls.length - 1);
    else idx = Math.max(idx - 1, 0);
    var id = slideEls[idx].getAttribute("data-slide");
    scrollToSlide(id);
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
      var text = bibtexBlock.textContent;
      navigator.clipboard.writeText(text).then(
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

  loadGitHubStats();
})();
