async function uploadAndProcess() {
    const outputDiv = document.getElementById('output');
    const loadingDiv = document.getElementById('loading-circle');

    // Clear previous output and show loading circle
    outputDiv.innerHTML = '';
    loadingDiv.style.display = 'block';

    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch('/process-pdf/', {
        method: 'POST',
        body: formData
    });

    const result = await response.json();
    
    // Hide loading circle and display result
    loadingDiv.style.display = 'none';
    displayResult(result);
}

function displayResult(data) {
    const outputDiv = document.getElementById('output');
    outputDiv.innerHTML = '<h2>Financial Report</h2>';
    
    for (const key in data) {
        outputDiv.innerHTML += `<div class="data-row"><span class="label">${key}:</span> <span class="value">${data[key]}</span></div>`;
    }
}



function showFileName() {
    const fileInput = document.getElementById('fileInput');
    const fileNameDisplay = document.getElementById('file-name');

    if (fileInput.files.length > 0) {
        fileNameDisplay.textContent = `Selected: ${fileInput.files[0].name}`;
    } else {
        fileNameDisplay.textContent = '';
    }
}

async function uploadAndProcess() {
    const outputDiv = document.getElementById('output');
    const loadingDiv = document.getElementById('loading-circle');
    const fileNameDisplay = document.getElementById('file-name');

    // Clear previous output, file name and show loading circle
    outputDiv.innerHTML = '';
    fileNameDisplay.textContent = '';
    loadingDiv.style.display = 'block';

    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch('/process-pdf/', {
        method: 'POST',
        body: formData
    });

    const result = await response.json();
    
    // Hide loading circle and display result
    loadingDiv.style.display = 'none';
    displayResult(result);
}

// ... rest of your JavaScript code for displayResult function
