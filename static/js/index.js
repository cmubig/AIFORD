(function () {
  const scrollBtn = document.getElementById('scrollToTopBtn');
  const copyBtn = document.getElementById('copyBibBtn');
  const bibtex = document.getElementById('bibtexCode');
  const resourceToggle = document.getElementById('resourceToggle');
  const resourceDropdown = document.getElementById('resourceDropdown');
  const resourceClose = document.getElementById('resourceClose');

  function setScrollBtnState() {
    if (!scrollBtn) return;
    if (window.scrollY > 320) {
      scrollBtn.classList.add('visible');
    } else {
      scrollBtn.classList.remove('visible');
    }
  }

  function toggleResources(forceOpen) {
    if (!resourceDropdown || !resourceToggle) return;
    const shouldOpen = typeof forceOpen === 'boolean'
      ? forceOpen
      : !resourceDropdown.classList.contains('show');

    resourceDropdown.classList.toggle('show', shouldOpen);
    resourceToggle.setAttribute('aria-expanded', shouldOpen ? 'true' : 'false');
  }

  function initScrollReveal() {
    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduceMotion) return;

    const selectors = [
      '.teaser-block .hero-body',
      '.section .title.is-3',
      '.section .abstract-text',
      '.section .metric-card',
      '.section .card-panel',
      '.section .contrib-card',
      '.section .table-wrap',
      '.section .table-hint',
      '.section .framed-figure',
      '.section .figure-note',
      '.section .mini-note',
      '.section .content'
    ];

    const targets = Array.from(
      new Set(selectors.flatMap(function (selector) {
        return Array.from(document.querySelectorAll(selector));
      }))
    );

    if (!targets.length) return;

    targets.forEach(function (el, index) {
      el.classList.add('reveal-item');
      el.style.setProperty('--reveal-delay', String((index % 5) * 45) + 'ms');
      const isInitiallyVisible = el.getBoundingClientRect().top <= window.innerHeight * 0.9;
      if (isInitiallyVisible) {
        el.classList.add('reveal-visible');
      }
    });

    const pending = targets.filter(function (el) {
      return el.classList.contains('reveal-item') && !el.classList.contains('reveal-visible');
    });

    if (!pending.length) return;

    if (!('IntersectionObserver' in window)) {
      pending.forEach(function (el) {
        el.classList.add('reveal-visible');
      });
      return;
    }

    const observer = new IntersectionObserver(function (entries, obs) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add('reveal-visible');
          obs.unobserve(entry.target);
        }
      });
    }, {
      threshold: 0.08,
      rootMargin: '0px 0px -8% 0px'
    });

    pending.forEach(function (el) {
      observer.observe(el);
    });
  }

  function initSectionSpy() {
    const navLinks = Array.from(document.querySelectorAll('.quick-nav a[href^="#"]'));
    if (!navLinks.length || !('IntersectionObserver' in window)) return;

    const sectionById = new Map();
    navLinks.forEach(function (link) {
      const id = link.getAttribute('href').slice(1);
      const section = document.getElementById(id);
      if (section) sectionById.set(id, section);
    });

    const activate = function (id) {
      navLinks.forEach(function (link) {
        link.classList.toggle('is-active', link.getAttribute('href') === '#' + id);
      });
    };

    const observer = new IntersectionObserver(function (entries) {
      const visible = entries
        .filter(function (entry) { return entry.isIntersecting; })
        .sort(function (a, b) { return b.intersectionRatio - a.intersectionRatio; });

      if (visible.length) {
        activate(visible[0].target.id);
      }
    }, {
      threshold: [0.2, 0.4, 0.6],
      rootMargin: '-20% 0px -55% 0px'
    });

    sectionById.forEach(function (section) {
      observer.observe(section);
    });
  }

  function initFigureSwipers() {
    const swipers = Array.from(document.querySelectorAll('.figure-swiper'));
    if (!swipers.length) return;

    swipers.forEach(function (swiper) {
      const viewport = swiper.querySelector('.figure-swiper-viewport');
      const track = swiper.querySelector('.figure-swiper-track');
      const slides = Array.from(swiper.querySelectorAll('.figure-swiper-slide'));
      const prevBtn = swiper.querySelector('.swiper-btn.prev');
      const nextBtn = swiper.querySelector('.swiper-btn.next');
      const dotsWrap = swiper.querySelector('.swiper-dots');
      if (!viewport || !track || slides.length === 0 || !dotsWrap) return;

      let index = 0;
      let touchStartX = null;

      const dots = slides.map(function (_, i) {
        const dot = document.createElement('button');
        dot.type = 'button';
        dot.className = 'swiper-dot';
        dot.setAttribute('aria-label', `Go to slide ${i + 1}`);
        dot.addEventListener('click', function () {
          index = i;
          render();
        });
        dotsWrap.appendChild(dot);
        return dot;
      });

      function syncViewportHeight() {
        const active = slides[index];
        if (!active) return;
        const height = active.offsetHeight;
        if (!height) return;
        viewport.style.height = `${height}px`;
      }

      function render() {
        track.style.transform = `translateX(-${index * 100}%)`;
        slides.forEach(function (slide, i) {
          slide.setAttribute('aria-hidden', i === index ? 'false' : 'true');
        });
        dots.forEach(function (dot, i) {
          dot.classList.toggle('is-active', i === index);
        });
        syncViewportHeight();
      }

      function goNext() {
        index = (index + 1) % slides.length;
        render();
      }

      function goPrev() {
        index = (index - 1 + slides.length) % slides.length;
        render();
      }

      if (nextBtn) nextBtn.addEventListener('click', goNext);
      if (prevBtn) prevBtn.addEventListener('click', goPrev);

      viewport.addEventListener('keydown', function (event) {
        if (event.key === 'ArrowRight') {
          event.preventDefault();
          goNext();
        } else if (event.key === 'ArrowLeft') {
          event.preventDefault();
          goPrev();
        }
      });

      viewport.addEventListener('touchstart', function (event) {
        touchStartX = event.changedTouches[0].clientX;
      }, { passive: true });

      viewport.addEventListener('touchend', function (event) {
        if (touchStartX === null) return;
        const delta = event.changedTouches[0].clientX - touchStartX;
        touchStartX = null;
        if (Math.abs(delta) < 35) return;
        if (delta < 0) {
          goNext();
        } else {
          goPrev();
        }
      }, { passive: true });

      slides.forEach(function (slide) {
        slide.querySelectorAll('img').forEach(function (img) {
          if (!img.complete) {
            img.addEventListener('load', syncViewportHeight);
          }
        });
      });

      window.addEventListener('resize', syncViewportHeight);

      render();
    });
  }

  async function copyBibTeX() {
    if (!copyBtn || !bibtex) return;
    const text = bibtex.textContent || '';

    try {
      await navigator.clipboard.writeText(text);
    } catch (error) {
      const ta = document.createElement('textarea');
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      document.body.removeChild(ta);
    }

    copyBtn.classList.add('copied');
    copyBtn.textContent = 'Copied';
    window.setTimeout(function () {
      copyBtn.classList.remove('copied');
      copyBtn.textContent = 'Copy';
    }, 1800);
  }

  if (scrollBtn) {
    scrollBtn.addEventListener('click', function () {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
  }

  if (copyBtn) {
    copyBtn.addEventListener('click', copyBibTeX);
  }

  if (resourceToggle) {
    resourceToggle.addEventListener('click', function () {
      toggleResources();
    });
  }

  if (resourceClose) {
    resourceClose.addEventListener('click', function () {
      toggleResources(false);
    });
  }

  document.addEventListener('click', function (event) {
    if (!resourceDropdown || !resourceToggle) return;

    const withinToggle = resourceToggle.contains(event.target);
    const withinDropdown = resourceDropdown.contains(event.target);
    if (!withinToggle && !withinDropdown) {
      toggleResources(false);
    }
  });

  document.addEventListener('keydown', function (event) {
    if (event.key === 'Escape') {
      toggleResources(false);
    }
  });

  window.addEventListener('scroll', setScrollBtnState);
  initScrollReveal();
  initSectionSpy();
  initFigureSwipers();
  setScrollBtnState();
})();
