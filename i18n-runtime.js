(function () {
  "use strict";

  var STORAGE_KEY = "echo-memory-lang";
  var NAV_MAP = {
    hero: "nav.home",
    overview: "nav.overview",
    design: "nav.design",
    checkpoints: "nav.checkpoints",
    evaluation: "nav.eval",
    evidence: "nav.evidence",
    findings: "nav.findings",
    updates: "nav.updates",
    bibtex: "nav.bibtex"
  };

  function detectDefaultLang() {
    var saved = localStorage.getItem(STORAGE_KEY);
    if (saved === "en" || saved === "zh") return saved;
    var lang = (navigator.language || "en").toLowerCase();
    return lang.indexOf("zh") === 0 ? "zh" : "en";
  }

  function dict(lang) {
    return (window.ECHO_I18N && window.ECHO_I18N[lang]) || {};
  }

  function t(lang, key) {
    var d = dict(lang);
    return Object.prototype.hasOwnProperty.call(d, key) ? d[key] : "";
  }

  function applyLang(lang) {
    var d = dict(lang);
    if (!Object.keys(d).length) return;

    document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";

    var descMeta = document.querySelector('meta[name="description"]');
    var pageTitleKey = document.body.getAttribute("data-title-key");
    if (pageTitleKey && d[pageTitleKey]) {
      document.title = d[pageTitleKey];
    } else if (d["meta.title"]) {
      document.title = d["meta.title"];
    }
    if (descMeta && d["meta.description"]) descMeta.setAttribute("content", d["meta.description"]);

    document.querySelectorAll("[data-i18n]").forEach(function (el) {
      var key = el.getAttribute("data-i18n");
      if (!key || !(key in d)) return;
      el.textContent = d[key];
    });

    document.querySelectorAll("[data-i18n-html]").forEach(function (el) {
      var key = el.getAttribute("data-i18n-html");
      if (!key || !(key in d)) return;
      el.innerHTML = d[key];
    });

    document.querySelectorAll("[data-i18n-attr]").forEach(function (el) {
      el.getAttribute("data-i18n-attr").split(";").forEach(function (pair) {
        var parts = pair.split(":");
        if (parts.length !== 2) return;
        var attr = parts[0].trim();
        var key = parts[1].trim();
        if (key in d) el.setAttribute(attr, d[key]);
      });
    });

    document.querySelectorAll("[data-nav]").forEach(function (el) {
      var navKey = NAV_MAP[el.getAttribute("data-nav")];
      if (navKey && d[navKey]) el.textContent = d[navKey];
    });

    document.querySelectorAll(".slide-dots button[data-slide]").forEach(function (btn) {
      var navKey = NAV_MAP[btn.getAttribute("data-slide")];
      if (navKey && d[navKey]) {
        btn.setAttribute("data-label", d[navKey]);
        btn.setAttribute("aria-label", d[navKey]);
      }
    });

    document.querySelectorAll(".qual-chip").forEach(function (chip, idx) {
      var keys = ["evidence.chip1", "evidence.chip2", "evidence.chip3"];
      if (keys[idx] && d[keys[idx]]) chip.textContent = d[keys[idx]];
    });

    var activeQual = document.querySelector(".qual-chip.is-active");
    var qualCaption = document.querySelector("[data-qual-caption]");
    if (activeQual && qualCaption && activeQual.dataset.captionKey && d[activeQual.dataset.captionKey]) {
      qualCaption.innerHTML = d[activeQual.dataset.captionKey] + ' <span class="zoom-hint">' + (d["zoom.hint"] || "Click to expand") + "</span>";
    } else if (qualCaption && d["evidence.cap1"] && document.querySelector(".qual-chip.is-active") === document.querySelector(".qual-chip")) {
      qualCaption.innerHTML = d["evidence.cap1"];
    }

    var toggle = document.getElementById("lang-toggle");
    if (toggle && d["lang.toggle"]) toggle.textContent = d["lang.toggle"];

    var langCurrent = document.querySelector("[data-lang-current]");
    if (langCurrent && d["lang.current"]) langCurrent.textContent = d["lang.current"];

    document.querySelectorAll("[data-copy-target]").forEach(function (btn) {
      if (!btn.classList.contains("is-copied") && d["bibtex.copy"]) {
        btn.textContent = d["bibtex.copy"];
      }
    });

    localStorage.setItem(STORAGE_KEY, lang);
    document.dispatchEvent(new CustomEvent("echo-lang-change", { detail: { lang: lang } }));
  }

  window.EchoI18n = {
    getLang: function () {
      return localStorage.getItem(STORAGE_KEY) || detectDefaultLang();
    },
    setLang: applyLang,
    t: function (key) {
      return t(window.EchoI18n.getLang(), key);
    }
  };

  document.addEventListener("DOMContentLoaded", function () {
    var lang = detectDefaultLang();
    applyLang(lang);

    var toggle = document.getElementById("lang-toggle");
    if (toggle) {
      toggle.addEventListener("click", function () {
        var next = window.EchoI18n.getLang() === "zh" ? "en" : "zh";
        applyLang(next);
      });
    }
  });
})();
