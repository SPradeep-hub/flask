  (function() {
    // modal handling exactly as before, no changes
    const watchBtn = document.getElementById('watchDemoBtn');
    const modal = document.getElementById('demoModal');
    const closeBtn = document.getElementById('closeModal');
    if (watchBtn && modal && closeBtn) {
      watchBtn.addEventListener('click', () => {
        modal.classList.remove('hidden');
      });
      closeBtn.addEventListener('click', () => {
        modal.classList.add('hidden');
      });
      window.addEventListener('click', (e) => {
        if (e.target === modal) {
          modal.classList.add('hidden');
        }
      });
    }
  })();