/* An illustrated, local walkthrough. No model calls, timers or external services. */
(() => {
  function initialise() {
    const demo = document.querySelector('[data-hl-demo]');
    if (!demo || demo.dataset.ready) return;
    demo.dataset.ready = 'true';
    const controls = document.querySelector('[data-hl-controls]');
    const buttons = controls.querySelectorAll('[data-hl-select]');
    controls.hidden = false;
    buttons.forEach(button => button.addEventListener('click', () => {
      const step = button.dataset.hlSelect;
      demo.querySelectorAll('[data-hl-step]').forEach(panel => {
        panel.hidden = panel.dataset.hlStep !== step;
      });
      demo.querySelector('[data-hl-output="chart"]').hidden = step === '2';
      demo.querySelector('[data-hl-output="code"]').hidden = step !== '2';
      demo.querySelector('[data-hl-insight]').hidden = step !== '1';
      demo.querySelector('[data-hl-caption]').hidden = step === '1';
      buttons.forEach(control => control.setAttribute('aria-pressed', String(control === button)));
    }));
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initialise);
  } else {
    initialise();
  }
})();
