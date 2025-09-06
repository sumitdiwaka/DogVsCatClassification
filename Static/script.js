

document.addEventListener('DOMContentLoaded', function() {
  // --- Element Selection ---
  const body = document.body;
  const form = document.getElementById('classifier-form');
  const fileInput = document.getElementById('fileInput');
  const dropArea = document.getElementById('dropArea');
  const dropText = dropArea.querySelector('.drop-text');
  const predictBtn = document.getElementById('predictBtn');
  const resultContainer = document.getElementById('resultContainer');
  const predictionResult = document.getElementById('predictionResult');
  const confidenceFill = document.getElementById('confidenceFill');
  const confidenceValue = document.getElementById('confidenceValue');
  const resultImage = document.getElementById('resultImage');

  // --- Event Listeners ---
  form.addEventListener('submit', handleFormSubmit);
  fileInput.addEventListener('change', handleFileSelect);
  setupDragAndDrop();

  // Make the drop area clickable
  dropArea.addEventListener('click', () => fileInput.click());

  // --- Drag & Drop ---
  function setupDragAndDrop() {
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
      dropArea.addEventListener(eventName, preventDefaults, false);
      document.body.addEventListener(eventName, preventDefaults, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
      dropArea.addEventListener(eventName, () => dropArea.classList.add('highlight'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
      dropArea.addEventListener(eventName, () => dropArea.classList.remove('highlight'), false);
    });

    dropArea.addEventListener('drop', handleDrop, false);

    function preventDefaults(e) {
      e.preventDefault();
      e.stopPropagation();
    }

    function handleDrop(e) {
      const files = e.dataTransfer.files;
      if (files.length) {
        fileInput.files = files;
        handleFileSelect(); // Trigger the preview
      }
    }
  }

  // --- File Handling & Preview ---
  function handleFileSelect() {
    const file = fileInput.files[0];
    if (!file) return;

    clearError();
    const existingPreview = dropArea.querySelector('.image-preview');
    if (existingPreview) existingPreview.remove();
    
    dropText.textContent = file.name;

    const reader = new FileReader();
    reader.onload = function(e) {
      const imgPreview = document.createElement('img');
      imgPreview.src = e.target.result;
      imgPreview.classList.add('image-preview');
      dropArea.insertBefore(imgPreview, dropText);
    };
    reader.readAsDataURL(file);

    resultContainer.style.display = 'none';
    predictBtn.querySelector('span').textContent = 'Predict';
  }

  // --- Form Submission & API Call ---
  async function handleFormSubmit(e) {
    e.preventDefault();
    if (!fileInput.files.length) {
      showFileError('Please select an image file first!');
      return;
    }

    showLoader();
    const file = fileInput.files[0];
    
    try {
      const result = await callClassificationAPI(file);
      if (result) {
        displayResult(result.label,result.confidence,  URL.createObjectURL(file));  //result.confidence result.label
      }
    } finally {
        hideLoader();
    }
  }
  
  async function callClassificationAPI(file) {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        body: formData
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.error || 'API request failed');
      }

      return result;
    } catch (error) {
      console.error('Error calling API:', error);
      alert(`Error processing image: ${error.message}`);
      return null;
    }
  }

  // --- UI State Functions (Loader, Result, Error) ---
  function showLoader() {
    predictBtn.disabled = true;
    predictBtn.querySelector('span').textContent = 'Predicting...';
    resultContainer.style.display = 'none';
  }

  function hideLoader() {
    predictBtn.disabled = false;
    predictBtn.querySelector('span').textContent = 'Predict Again';
  }

  function displayResult(prediction, confidence, imageUrl) {
    predictionResult.textContent = prediction;
    confidenceFill.style.width = `${confidence * 100}%`;
    confidenceValue.textContent = `${(confidence * 100).toFixed(1)}% confidence`;
    resultImage.src = imageUrl;

    const safeClassName = prediction.toLowerCase().replace(/\s+/g, '-').replace(/[^a-z0-9-]/g, '');

    predictionResult.className = 'prediction-text';
    if (safeClassName) {
      predictionResult.classList.add(safeClassName);
    }

    resultContainer.style.display = 'block';
    resultContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function showFileError(message) {
    dropArea.classList.add('error');
    const errorMsg = dropArea.querySelector('.error-message') || document.createElement('div');
    errorMsg.className = 'error-message';
    errorMsg.textContent = message;
    if (!dropArea.querySelector('.error-message')) {
      dropArea.appendChild(errorMsg);
    }
    dropArea.style.animation = 'shake 0.5s';
    dropArea.addEventListener('animationend', () => {
        dropArea.style.animation = '';
    }, { once: true });
  }
  
  function clearError() {
      dropArea.classList.remove('error');
      const errorMsg = dropArea.querySelector('.error-message');
      if (errorMsg) errorMsg.remove();
  }
});


