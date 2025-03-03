function showLoadingSpinner() {
    document.getElementById('loading-overlay').style.display = 'flex';
  }
  
  const form = document.querySelector('form');
  if (form) {
    form.addEventListener('submit', function() {
      showLoadingSpinner();
    });
  }

window.onload = function() {
  var audio = document.querySelector('audio');
  audio.play();
};

async function setupWebcam() {
  const video = document.getElementById('webcam');
  try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      video.srcObject = stream;
  } catch (error) {
      console.error('Error accessing webcam: ', error);
  }
}

window.addEventListener('load', setupWebcam);