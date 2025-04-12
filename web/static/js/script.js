document.addEventListener('DOMContentLoaded', function() {
    // Tab switching functionality
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            // Remove active class from all buttons and panes
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabPanes.forEach(pane => pane.classList.remove('active'));
            
            // Add active class to the clicked button and corresponding pane
            button.classList.add('active');
            const tabId = button.dataset.tab;
            document.getElementById(tabId).classList.add('active');
        });
    });
    
    // Load initial data
    loadDataSample();
    loadMetrics();
    
    // Form submissions
    setupPredictionForm();
    setupTrainingForm();
    setupExperimentForm();
    
    // Tag input functionality for experiment tab
    setupTagInputs();
});

// Show or hide loading overlay
function toggleLoading(show, message = 'Processing...') {
    const loadingOverlay = document.getElementById('loading-overlay');
    const loadingMessage = document.getElementById('loading-message');
    
    if (show) {
        loadingMessage.textContent = message;
        loadingOverlay.classList.remove('hidden');
    } else {
        loadingOverlay.classList.add('hidden');
    }
}

// Display an error message
function showError(containerId, message) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    container.classList.add('error');
    
    const p = document.createElement('p');
    p.textContent = message;
    container.appendChild(p);
}

// Display a success message
function showSuccess(containerId, message) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    container.classList.add('success');
    
    const p = document.createElement('p');
    p.textContent = message;
    container.appendChild(p);
}

// Format a number as a percentage
function formatPercent(value) {
    return (value * 100).toFixed(2) + '%';
}

// Setup prediction form submission
function setupPredictionForm() {
    const form = document.getElementById('prediction-form');
    
    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        
        const text = document.getElementById('prediction-text').value.trim();
        const modelType = document.getElementById('prediction-model-type').value;
        
        if (!text) {
            showError('prediction-message', 'Please enter text to classify.');
            return;
        }
        
        toggleLoading(true, 'Making prediction...');
        
        try {
            const formData = new FormData();
            formData.append('text', text);
            formData.append('model_type', modelType);
            
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.success) {
                showSuccess('prediction-message', data.message);
                displayPredictionResult(data.result, data.prediction);
            } else {
                showError('prediction-message', data.message);
                document.getElementById('prediction-box').classList.add('hidden');
            }
        } catch (error) {
            showError('prediction-message', 'Error connecting to server. Please try again.');
            console.error('Error:', error);
        } finally {
            toggleLoading(false);
        }
    });
}

// Display the prediction result visualization
function displayPredictionResult(result, prediction) {
    const predictionBox = document.getElementById('prediction-box');
    const predictionLabel = predictionBox.querySelector('.prediction-label');
    const meterBar = predictionBox.querySelector('.meter-bar');
    
    predictionBox.classList.remove('hidden', 'spam', 'not-spam');
    meterBar.classList.remove('spam', 'not-spam');
    
    if (result === 'SPAM') {
        predictionBox.classList.add('spam');
        meterBar.classList.add('spam');
        predictionLabel.textContent = '✖ Classified as SPAM';
    } else {
        predictionBox.classList.add('not-spam');
        meterBar.classList.add('not-spam');
        predictionLabel.textContent = '✓ Classified as NOT SPAM';
    }
    
    // Set confidence meter (this is a placeholder as we don't have actual confidence scores)
    // In a real app, you might want to use model probabilities here
    meterBar.style.width = prediction === 1 ? '80%' : '80%';
}

// Setup training form submission
function setupTrainingForm() {
    const form = document.getElementById('training-form');
    
    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        
        const modelType = document.getElementById('training-model-type').value;
        const epochs = document.getElementById('training-epochs').value;
        const batchSize = document.getElementById('training-batch-size').value;
        const hiddenLayers = document.getElementById('training-hidden-layers').value;
        
        toggleLoading(true, 'Training model... This may take a while.');
        
        try {
            const formData = new FormData();
            formData.append('model_type', modelType);
            formData.append('epochs', epochs);
            formData.append('batch_size', batchSize);
            formData.append('hidden_layers', hiddenLayers);
            
            const response = await fetch('/train', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.success) {
                showSuccess('training-message', data.message);
                displayTrainingMetrics(data.metrics);
            } else {
                showError('training-message', data.message);
                document.getElementById('training-metrics').classList.add('hidden');
            }
        } catch (error) {
            showError('training-message', 'Error connecting to server. Please try again.');
            console.error('Error:', error);
        } finally {
            toggleLoading(false);
        }
    });
}

// Display training metrics
function displayTrainingMetrics(metrics) {
    const metricsContainer = document.getElementById('training-metrics');
    metricsContainer.classList.remove('hidden');
    
    // Update metric values
    for (const [key, value] of Object.entries(metrics)) {
        const valueElement = metricsContainer.querySelector(`.${key}`);
        if (valueElement) {
            valueElement.textContent = formatPercent(value);
        }
    }
}

// Setup experiment form submission
function setupExperimentForm() {
    const form = document.getElementById('experiment-form');
    
    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        
        toggleLoading(true, 'Running experiments... This may take a while.');
        
        try {
            const formData = new FormData(form);
            
            const response = await fetch('/run_experiment', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.success) {
                showSuccess('experiment-message', data.message);
                displayBestModel(data.best_model);
            } else {
                showError('experiment-message', data.message);
                document.getElementById('best-model').classList.add('hidden');
            }
        } catch (error) {
            showError('experiment-message', 'Error connecting to server. Please try again.');
            console.error('Error:', error);
        } finally {
            toggleLoading(false);
        }
    });
}

// Display best model from auto-experiment
function displayBestModel(bestModel) {
    const bestModelContainer = document.getElementById('best-model');
    bestModelContainer.classList.remove('hidden');
    
    // Update best model details
    bestModelContainer.querySelector('.model-type').textContent = bestModel.model_type;
    bestModelContainer.querySelector('.epochs').textContent = bestModel.epochs;
    bestModelContainer.querySelector('.batch-size').textContent = bestModel.batch_size;
    bestModelContainer.querySelector('.accuracy').textContent = 
        formatPercent(bestModel.metrics.accuracy);
}

// Load sample data
async function loadDataSample() {
    try {
        const response = await fetch('/get_data_sample');
        const data = await response.json();
        
        if (data.success) {
            displayDataSample(data.data);
        } else {
            const tableBody = document.querySelector('#data-table tbody');
            tableBody.innerHTML = `<tr><td colspan="2">${data.message}</td></tr>`;
        }
    } catch (error) {
        console.error('Error loading data sample:', error);
        const tableBody = document.querySelector('#data-table tbody');
        tableBody.innerHTML = '<tr><td colspan="2">Failed to load data. Please try refreshing the page.</td></tr>';
    }
}

// Display sample data in the table
function displayDataSample(data) {
    const tableBody = document.querySelector('#data-table tbody');
    tableBody.innerHTML = '';
    
    for (let i = 0; i < data.messages.length; i++) {
        const row = document.createElement('tr');
        
        const messageCell = document.createElement('td');
        messageCell.textContent = data.messages[i];
        row.appendChild(messageCell);
        
        const labelCell = document.createElement('td');
        const labelSpan = document.createElement('span');
        labelSpan.className = data.labels[i] === 1 ? 'spam-label' : 'not-spam-label';
        labelSpan.textContent = data.labels[i] === 1 ? 'SPAM' : 'NOT SPAM';
        labelCell.appendChild(labelSpan);
        row.appendChild(labelCell);
        
        tableBody.appendChild(row);
    }
}

// Load metrics for the metrics tab
async function loadMetrics() {
    try {
        const response = await fetch('/get_metrics');
        const data = await response.json();
        
        if (data.success) {
            displayMetricsInfo(data);
            displayMetricsChart(data.metrics);
            displayMetricsCards(data.metrics);
        }
    } catch (error) {
        console.error('Error loading metrics:', error);
    }
}

// Display model info in the metrics tab
function displayMetricsInfo(data) {
    document.getElementById('info-model-type').textContent = data.model_type;
    document.getElementById('info-timestamp').textContent = data.timestamp;
    document.getElementById('info-epochs').textContent = data.epochs;
    document.getElementById('info-batch-size').textContent = data.batch_size;
}

// Display metrics chart
function displayMetricsChart(metrics) {
    const ctx = document.getElementById('metrics-chart').getContext('2d');
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: Object.keys(metrics),
            datasets: [{
                label: 'Performance Metrics',
                data: Object.values(metrics),
                backgroundColor: [
                    'rgba(74, 110, 245, 0.7)',
                    'rgba(28, 167, 69, 0.7)',
                    'rgba(220, 53, 69, 0.7)',
                    'rgba(255, 193, 7, 0.7)'
                ],
                borderColor: [
                    'rgba(74, 110, 245, 1)',
                    'rgba(28, 167, 69, 1)',
                    'rgba(220, 53, 69, 1)',
                    'rgba(255, 193, 7, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1,
                    ticks: {
                        callback: function(value) {
                            return formatPercent(value);
                        }
                    }
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return formatPercent(context.parsed.y);
                        }
                    }
                }
            }
        }
    });
}

// Display metrics in cards
function displayMetricsCards(metrics) {
    document.getElementById('metric-accuracy').textContent = formatPercent(metrics.accuracy || 0);
    document.getElementById('metric-precision').textContent = formatPercent(metrics.precision || 0);
    document.getElementById('metric-recall').textContent = formatPercent(metrics.recall || 0);
    document.getElementById('metric-f1').textContent = formatPercent(metrics.f1_score || 0);
}

// Setup tag inputs for the experiment tab
function setupTagInputs() {
    setupTagInput('epoch-input', 'add-epoch', 'epoch-tags', 'epoch_options[]');
    setupTagInput('batch-input', 'add-batch', 'batch-tags', 'batch_sizes[]');
    
    // Setup remove tag buttons
    document.querySelectorAll('.remove-tag').forEach(button => {
        button.addEventListener('click', function() {
            const tag = this.parentElement;
            const value = tag.textContent.replace('×', '').trim();
            const tagsContainer = tag.parentElement;
            const inputName = tagsContainer.id === 'epoch-tags' ? 'epoch_options[]' : 'batch_sizes[]';
            
            // Remove the hidden input with this value
            const hiddenInput = tagsContainer.querySelector(`input[name="${inputName}"][value="${value}"]`);
            if (hiddenInput) {
                hiddenInput.remove();
            }
            
            // Remove the tag
            tag.remove();
        });
    });
}

// Setup tag input functionality
function setupTagInput(inputId, buttonId, tagsContainerId, hiddenInputName) {
    const input = document.getElementById(inputId);
    const addButton = document.getElementById(buttonId);
    const tagsContainer = document.getElementById(tagsContainerId);
    
    addButton.addEventListener('click', function() {
        const value = input.value.trim();
        
        if (value && !isNaN(value)) {
            // Check if tag already exists
            const existingTags = Array.from(tagsContainer.querySelectorAll('.tag')).map(tag => 
                tag.textContent.replace('×', '').trim()
            );
            
            if (!existingTags.includes(value)) {
                // Create the tag
                const tag = document.createElement('span');
                tag.className = 'tag';
                tag.textContent = value;
                
                // Add remove button
                const removeButton = document.createElement('button');
                removeButton.className = 'remove-tag';
                removeButton.textContent = '×';
                removeButton.addEventListener('click', function() {
                    // Remove the hidden input
                    const hiddenInput = tagsContainer.querySelector(`input[name="${hiddenInputName}"][value="${value}"]`);
                    if (hiddenInput) {
                        hiddenInput.remove();
                    }
                    
                    // Remove the tag
                    tag.remove();
                });
                
                tag.appendChild(removeButton);
                
                // Add hidden input
                const hiddenInput = document.createElement('input');
                hiddenInput.type = 'hidden';
                hiddenInput.name = hiddenInputName;
                hiddenInput.value = value;
                
                // Add to the DOM
                tagsContainer.appendChild(tag);
                tagsContainer.appendChild(hiddenInput);
                
                // Clear the input
                input.value = '';
            }
        }
    });
    
    // Also allow pressing Enter to add a tag
    input.addEventListener('keypress', function(event) {
        if (event.key === 'Enter') {
            event.preventDefault();
            addButton.click();
        }
    });
}