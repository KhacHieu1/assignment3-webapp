(function () {
  "use strict";

  function qs(sel, root) {
    return (root || document).querySelector(sel);
  }
  function qsa(sel, root) {
    return Array.from((root || document).querySelectorAll(sel));
  }

  /* Sticky header shadow */
  var header = qs("[data-sticky-header]");
  if (header) {
    var onScroll = function () {
      header.classList.toggle("is-scrolled", window.scrollY > 8);
    };
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
  }

  /* Mobile nav */
  var navToggle = qs("[data-nav-toggle]");
  var primaryNav = qs("[data-primary-nav]");
  if (navToggle && primaryNav) {
    navToggle.addEventListener("click", function () {
      var open = primaryNav.classList.toggle("is-open");
      navToggle.setAttribute("aria-expanded", open ? "true" : "false");
    });
  }

  /* Brand dropdown */
  qsa("[data-dropdown-trigger]").forEach(function (btn) {
    var wrap = btn.closest(".nav-dropdown");
    if (!wrap) return;
    btn.addEventListener("click", function (e) {
      e.stopPropagation();
      var isOpen = wrap.classList.toggle("is-open");
      btn.setAttribute("aria-expanded", isOpen ? "true" : "false");
    });
  });
  document.addEventListener("click", function () {
    qsa(".nav-dropdown.is-open").forEach(function (w) {
      w.classList.remove("is-open");
      var b = qs("[data-dropdown-trigger]", w);
      if (b) b.setAttribute("aria-expanded", "false");
    });
  });

  /* Hero carousel */
  var slides = qsa("[data-hero-slide]");
  var dots = qsa("[data-hero-dot]");
  if (slides.length && dots.length) {
    var idx = 0;
    function show(i) {
      idx = (i + slides.length) % slides.length;
      slides.forEach(function (s, j) {
        s.classList.toggle("is-active", j === idx);
      });
      dots.forEach(function (d, j) {
        d.classList.toggle("is-active", j === idx);
        d.setAttribute("aria-selected", j === idx ? "true" : "false");
      });
    }
    dots.forEach(function (d) {
      d.addEventListener("click", function () {
        show(Number(d.getAttribute("data-hero-dot")) || 0);
      });
    });
    setInterval(function () {
      show(idx + 1);
    }, 7000);
  }

  /* Search suggest + clear */
  var form = qs("[data-search-form]");
  if (form) {
    var input = qs("[data-search-input]", form);
    var panel = qs("[data-suggest-panel]", form);
    var clearBtn = qs("[data-search-clear]", form);
    var urlBase = input && input.getAttribute("data-suggest-url");
    var tmo = null;

    function setClearVisible() {
      if (!clearBtn || !input) return;
      clearBtn.hidden = !input.value.length;
    }
    if (input) {
      input.addEventListener("input", setClearVisible);
      setClearVisible();
    }
    if (clearBtn && input) {
      clearBtn.addEventListener("click", function () {
        input.value = "";
        setClearVisible();
        input.focus();
        if (panel) panel.hidden = true;
      });
    }

    function renderSuggest(items) {
      if (!panel) return;
      panel.innerHTML = "";
      items.forEach(function (text, i) {
        var b = document.createElement("button");
        b.type = "button";
        b.className = "suggest-item" + (i === 0 ? " is-active" : "");
        b.textContent = text;
        b.addEventListener("mousedown", function (e) {
          e.preventDefault();
          input.value = text;
          form.submit();
        });
        panel.appendChild(b);
      });
      panel.hidden = items.length === 0;
    }

    if (input && panel && urlBase) {
      input.addEventListener("input", function () {
        clearTimeout(tmo);
        var q = input.value.trim();
        if (q.length < 1) {
          panel.hidden = true;
          return;
        }
        tmo = setTimeout(function () {
          fetch(urlBase + "?q=" + encodeURIComponent(q), { headers: { Accept: "application/json" } })
            .then(function (r) {
              return r.json();
            })
            .then(renderSuggest)
            .catch(function () {
              panel.hidden = true;
            });
        }, 200);
      });
      input.addEventListener("blur", function () {
        setTimeout(function () {
          panel.hidden = true;
        }, 150);
      });
      input.addEventListener("focus", function () {
        if (panel.childElementCount) panel.hidden = false;
      });
    }
  }

  /* Auto-dismiss server toasts + toast root */
  var toastRoot = qs("#toast-root");
  function pushToast(message, kind) {
    if (!toastRoot) return;
    var el = document.createElement("div");
    var k = kind || "success";
    el.className = "toast toast-" + k;
    el.textContent = message;
    toastRoot.appendChild(el);
    setTimeout(function () {
      el.remove();
    }, 4500);
  }

  qsa("[data-toast]").forEach(function (t) {
    pushToast(t.textContent, t.classList.contains("toast-success") ? "success" : "info");
    t.remove();
  });

  /* Results filters */
  var grid = qs("[data-results-grid]");
  var jsonEl = qs("#search-results-json");
  if (grid && jsonEl) {
    var data = [];
    try {
      data = JSON.parse(jsonEl.textContent || "[]");
    } catch (e) {
      data = [];
    }
    var cards = qsa("[data-card]", grid);
    var brandBox = qs("[data-filter-brand]");
    var ratingSel = qs("[data-filter-rating]");
    var sortSel = qs("[data-sort-results]");
    var resetBtns = qsa("[data-filter-reset]");
    var loadingBar = qs("[data-loading-bar]");
    var noMatch = qs("[data-no-match-msg]");

    var brands = Array.from(
      new Set(
        data
          .map(function (r) {
            return r.brand_name;
          })
          .filter(Boolean)
      )
    ).sort();

    if (brandBox) {
      brands.slice(0, 24).forEach(function (b) {
        var id = "b_" + b.replace(/\W+/g, "_").slice(0, 40);
        var lab = document.createElement("label");
        lab.className = "filter-check";
        lab.innerHTML =
          '<input type="checkbox" value="' +
          b.replace(/"/g, "&quot;") +
          '" data-brand-cb> ' +
          "<span>" +
          b +
          "</span>";
        brandBox.appendChild(lab);
      });
    }

    function selectedBrands() {
      return qsa("[data-brand-cb]", brandBox || document)
        .filter(function (c) {
          return c.checked;
        })
        .map(function (c) {
          return c.value;
        });
    }

    function apply() {
      if (loadingBar) {
        loadingBar.hidden = false;
      }
      window.setTimeout(function () {
        var minRating = ratingSel ? Number(ratingSel.value || 0) : 0;
        var sBrands = selectedBrands();
        var visible = [];

        cards.forEach(function (card) {
          var brand = card.getAttribute("data-brand") || "";
          var rating = parseFloat(card.getAttribute("data-rating") || "0");
          var okBrand = sBrands.length === 0 || sBrands.indexOf(brand) !== -1;
          var okRate = !minRating || rating >= minRating;
          var show = okBrand && okRate;
          card.classList.toggle("is-hidden", !show);
          if (show) visible.push(card);
        });

        var sort = sortSel ? sortSel.value : "relevance";
        visible.sort(function (a, b) {
          if (sort === "rating-desc") {
            return parseFloat(b.getAttribute("data-rating")) - parseFloat(a.getAttribute("data-rating"));
          }
          if (sort === "price-asc") {
            return parseFloat(a.getAttribute("data-price")) - parseFloat(b.getAttribute("data-price"));
          }
          if (sort === "reviews") {
            return a.getAttribute("data-title").length - b.getAttribute("data-title").length;
          }
          return parseFloat(b.getAttribute("data-relevance")) - parseFloat(a.getAttribute("data-relevance"));
        });
        visible.forEach(function (c) {
          grid.appendChild(c);
        });

        if (loadingBar) loadingBar.hidden = true;
        if (noMatch) {
          noMatch.hidden = visible.length > 0;
        }
      }, 180);
    }

    qsa("[data-brand-cb]").forEach(function (c) {
      return c.addEventListener("change", apply);
    });
    if (ratingSel) ratingSel.addEventListener("change", apply);
    if (sortSel) sortSel.addEventListener("change", apply);
    resetBtns.forEach(function (b) {
      b.addEventListener("click", function () {
        qsa("[data-brand-cb]").forEach(function (c) {
          c.checked = false;
        });
        if (ratingSel) ratingSel.value = "0";
        if (sortSel) sortSel.value = "relevance";
        apply();
      });
    });
  }

  /* Create review: char count, mock ML, override panel */
  var reviewBody = qs("[data-review-body]");
  var charCount = qs("[data-char-count]");
  if (reviewBody && charCount) {
    function upd() {
      charCount.textContent = String(reviewBody.value.length);
    }
    reviewBody.addEventListener("input", upd);
    upd();
  }

  var predBox = qs("[data-prediction-box]");
  if (predBox) {
    var ph = qs("[data-prediction-placeholder]", predBox);
    var res = qs("[data-prediction-result]", predBox);
    var load = qs("[data-prediction-loading]", predBox);
    var lbl = qs("[data-prediction-label]", predBox);
    var conf = qs("[data-prediction-confidence]", predBox);
    var expl = qs("[data-prediction-explainer]", predBox);
    var t2 = null;

    function mockPredict(text) {
      var lower = text.toLowerCase();
      var pos = ["love", "great", "best", "worth", "buy", "recommend", "amazing", "good", "perfect"];
      var neg = ["waste", "bad", "not worth", "poor", "disappointed", "worst", "return"];
      var score = 55;
      pos.forEach(function (w) {
        if (lower.indexOf(w) !== -1) score += 8;
      });
      neg.forEach(function (w) {
        if (lower.indexOf(w) !== -1) score -= 12;
      });
      score = Math.max(52, Math.min(96, score + (text.length % 9)));
      var buyer = score >= 68;
      return {
        buyer: buyer,
        confidence: score,
        expl: buyer
          ? "Language resembles purchase-ready reviewers in the training slice."
          : "Sceptical or negative cues reduce predicted buyer likelihood.",
      };
    }

    if (reviewBody) {
      reviewBody.addEventListener("input", function () {
        clearTimeout(t2);
        var txt = reviewBody.value.trim();
        if (txt.length < 12) {
          if (ph) ph.hidden = false;
          if (res) res.hidden = true;
          if (load) load.hidden = true;
          return;
        }
        if (load) load.hidden = false;
        if (res) res.hidden = true;
        if (ph) ph.hidden = true;
        t2 = setTimeout(function () {
          var out = mockPredict(txt);
          if (load) load.hidden = true;
          if (res) res.hidden = false;
          if (lbl) lbl.textContent = out.buyer ? "Likely buyer" : "Unlikely buyer";
          if (conf) conf.textContent = out.confidence + "%";
          if (expl) expl.textContent = out.expl;
        }, 450);
      });
    }
  }

  qsa("[data-label-mode]").forEach(function (r) {
    r.addEventListener("change", function () {
      var panel = qs("[data-override-panel]");
      if (!panel) return;
      var show = r.value === "override" && r.checked;
      panel.hidden = !show;
    });
  });

  var reviewForm = qs("[data-review-form]");
  if (reviewForm) {
    reviewForm.addEventListener("submit", function (e) {
      var body = qs("[data-review-body]", reviewForm);
      if (body && body.value.trim().length < 20) {
        e.preventDefault();
        pushToast("Review description must be at least 20 characters.", "error");
        return;
      }
      pushToast("Submitting review…", "info");
    });
  }

  qsa("[data-scroll-to]").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var sel = btn.getAttribute("data-scroll-to");
      var el = sel && document.querySelector(sel);
      if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  });
})();
