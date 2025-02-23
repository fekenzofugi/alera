function showLoadingSpinner() {
    document.getElementById('loading-overlay').style.display = 'flex';
  }
  
  const form = document.querySelector('form');
  if (form) {
    form.addEventListener('submit', function() {
      showLoadingSpinner();
    });
  }