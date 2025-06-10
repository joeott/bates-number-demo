/**
 * Legal Document Retriever - Frontend JavaScript
 * 
 * This module handles all frontend interactions for the retrieval UI,
 * including query submission, result display, and history management.
 * 
 * Features:
 * - Input validation and sanitization
 * - Keyboard navigation support
 * - ARIA live regions for accessibility
 * - XSS prevention
 * - Rate limit handling
 */

// Configuration
const API_BASE_URL = 'http://localhost:8003';
const STORAGE_KEY_HISTORY = 'legal_retriever_history';
const MAX_HISTORY_ITEMS = 10;

// State
let isProcessing = false;
let queryHistory = [];

// DOM Elements
const elements = {
    queryForm: document.getElementById('queryForm'),
    queryInput: document.getElementById('queryInput'),
    maxIterations: document.getElementById('maxIterations'),
    submitBtn: document.getElementById('submitBtn'),
    charCount: document.getElementById('charCount'),
    apiStatus: document.getElementById('apiStatus'),
    errorAlert: document.getElementById('errorAlert'),
    errorMessage: document.getElementById('errorMessage'),
    resultsSection: document.getElementById('resultsSection'),
    answerContent: document.getElementById('answerContent'),
    resultMeta: document.getElementById('resultMeta'),
    historySection: document.getElementById('historySection'),
    historyList: document.getElementById('historyList')
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

/**
 * Initialize the application
 */
async function initializeApp() {
    // Check API health
    await checkAPIHealth();
    
    // Load query history
    loadHistory();
    
    // Setup event listeners
    setupEventListeners();
    
    // Focus on query input
    elements.queryInput.focus();
    
    // Setup keyboard navigation
    setupKeyboardNavigation();
}

/**
 * Setup all event listeners
 */
function setupEventListeners() {
    // Form submission
    elements.queryForm.addEventListener('submit', handleQuerySubmit);
    
    // Character counter
    elements.queryInput.addEventListener('input', updateCharCount);
    
    // Initial character count
    updateCharCount();
    
    // Form validation
    elements.queryForm.addEventListener('invalid', handleInvalidForm, true);
    
    // Escape key to clear errors
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            hideError();
        }
    });
}

/**
 * Check API health status
 */
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        
        if (data.status === 'healthy' && data.agent_loaded) {
            elements.apiStatus.textContent = 'Connected';
            elements.apiStatus.className = 'stat-value status-healthy';
        } else {
            elements.apiStatus.textContent = 'Degraded';
            elements.apiStatus.className = 'stat-value status-error';
        }
    } catch (error) {
        elements.apiStatus.textContent = 'Offline';
        elements.apiStatus.className = 'stat-value status-error';
        console.error('API health check failed:', error);
    }
}

/**
 * Handle query form submission
 */
async function handleQuerySubmit(event) {
    event.preventDefault();
    
    if (isProcessing) return;
    
    // Validate form
    if (!elements.queryForm.checkValidity()) {
        elements.queryForm.reportValidity();
        return;
    }
    
    const query = sanitizeInput(elements.queryInput.value.trim());
    if (!query) {
        showError('Please enter a query');
        elements.queryInput.focus();
        return;
    }
    
    // Additional validation
    if (query.length < 3) {
        showError('Query must be at least 3 characters long');
        elements.queryInput.focus();
        return;
    }
    
    // Hide previous results/errors
    hideError();
    elements.resultsSection.classList.add('hidden');
    
    // Start processing
    setProcessingState(true);
    
    try {
        // Prepare request
        const requestBody = {
            query: query,
            request_id: generateRequestId()
        };
        
        // Add max iterations if specified
        const maxIterations = elements.maxIterations.value;
        if (maxIterations) {
            requestBody.max_iterations = parseInt(maxIterations);
        }
        
        // Make API request
        const response = await fetch(`${API_BASE_URL}/api/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody)
        });
        
        // Handle response
        const data = await response.json();
        
        if (!response.ok) {
            // Handle rate limiting
            if (response.status === 429) {
                const retryAfter = response.headers.get('Retry-After') || '60';
                throw new Error(`Rate limit exceeded. Please try again in ${retryAfter} seconds.`);
            }
            throw new Error(data.detail || data.error || `Server error: ${response.status}`);
        }
        
        if (data.success && data.answer) {
            displayResult(data);
            addToHistory(query, data.answer);
        } else {
            throw new Error(data.error || 'No answer received');
        }
        
    } catch (error) {
        console.error('Query failed:', error);
        showError(error.message || 'Failed to process query. Please try again.');
    } finally {
        setProcessingState(false);
    }
}

/**
 * Display query result
 */
function displayResult(data) {
    // Show results section
    elements.resultsSection.classList.remove('hidden');
    
    // Display answer
    elements.answerContent.textContent = data.answer;
    
    // Display metadata
    const metadata = [];
    if (data.processing_time) {
        metadata.push(`Processed in ${data.processing_time.toFixed(1)}s`);
    }
    if (data.metadata?.iterations_used) {
        metadata.push(`${data.metadata.iterations_used} iterations`);
    }
    if (data.metadata?.facts_found) {
        metadata.push(`${data.metadata.facts_found} facts found`);
    }
    
    elements.resultMeta.textContent = metadata.join(' • ');
    
    // Scroll to results
    elements.resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/**
 * Set processing state
 */
function setProcessingState(processing) {
    isProcessing = processing;
    
    if (processing) {
        elements.submitBtn.disabled = true;
        elements.submitBtn.querySelector('.btn-text').textContent = 'Searching...';
        elements.submitBtn.querySelector('.spinner').classList.remove('hidden');
        elements.queryInput.disabled = true;
        elements.maxIterations.disabled = true;
    } else {
        elements.submitBtn.disabled = false;
        elements.submitBtn.querySelector('.btn-text').textContent = 'Search Documents';
        elements.submitBtn.querySelector('.spinner').classList.add('hidden');
        elements.queryInput.disabled = false;
        elements.maxIterations.disabled = false;
    }
}

/**
 * Show error message
 */
function showError(message) {
    elements.errorMessage.textContent = message;
    elements.errorAlert.classList.remove('hidden');
    
    // Announce to screen readers
    elements.errorAlert.setAttribute('aria-live', 'assertive');
    
    // Focus management for accessibility
    elements.errorAlert.focus();
}

/**
 * Hide error message
 */
function hideError() {
    elements.errorAlert.classList.add('hidden');
    elements.errorAlert.setAttribute('aria-live', 'off');
}

/**
 * Update character count
 */
function updateCharCount() {
    const length = elements.queryInput.value.length;
    const maxLength = elements.queryInput.getAttribute('maxlength') || 1000;
    elements.charCount.textContent = `${length} / ${maxLength}`;
    
    // Change color when near limit
    if (length > maxLength * 0.9) {
        elements.charCount.style.color = 'var(--color-error)';
    } else {
        elements.charCount.style.color = 'var(--color-text-muted)';
    }
}

/**
 * Add query to history
 */
function addToHistory(query, answer) {
    const historyItem = {
        query: query,
        answer: answer,
        timestamp: new Date().toISOString()
    };
    
    // Add to beginning of array
    queryHistory.unshift(historyItem);
    
    // Limit history size
    if (queryHistory.length > MAX_HISTORY_ITEMS) {
        queryHistory = queryHistory.slice(0, MAX_HISTORY_ITEMS);
    }
    
    // Save and update display
    saveHistory();
    displayHistory();
}

/**
 * Load history from localStorage
 */
function loadHistory() {
    try {
        const saved = localStorage.getItem(STORAGE_KEY_HISTORY);
        if (saved) {
            queryHistory = JSON.parse(saved);
        }
    } catch (error) {
        console.error('Failed to load history:', error);
        queryHistory = [];
    }
    
    displayHistory();
}

/**
 * Save history to localStorage
 */
function saveHistory() {
    try {
        localStorage.setItem(STORAGE_KEY_HISTORY, JSON.stringify(queryHistory));
    } catch (error) {
        console.error('Failed to save history:', error);
    }
}

/**
 * Display history items
 */
function displayHistory() {
    if (queryHistory.length === 0) {
        elements.historyList.innerHTML = '<p class="empty-state">No queries yet. Ask a question to get started!</p>';
        return;
    }
    
    elements.historyList.innerHTML = queryHistory.map((item, index) => `
        <div class="history-item" onclick="loadHistoryItem(${index})">
            <div class="history-query">${escapeHtml(truncate(item.query, 100))}</div>
            <div class="history-time">${formatRelativeTime(item.timestamp)}</div>
        </div>
    `).join('');
}

/**
 * Load a history item
 */
window.loadHistoryItem = function(index) {
    const item = queryHistory[index];
    if (item) {
        elements.queryInput.value = item.query;
        updateCharCount();
        displayResult({
            answer: item.answer,
            query: item.query,
            success: true
        });
    }
};

/**
 * Clear history
 */
window.clearHistory = function() {
    if (confirm('Are you sure you want to clear all query history?')) {
        queryHistory = [];
        saveHistory();
        displayHistory();
    }
};

/**
 * Generate request ID
 */
function generateRequestId() {
    return `web-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Format relative time
 */
function formatRelativeTime(timestamp) {
    const date = new Date(timestamp);
    const now = new Date();
    const diff = now - date;
    
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);
    
    if (minutes < 1) return 'Just now';
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    if (days < 7) return `${days}d ago`;
    
    return date.toLocaleDateString();
}

/**
 * Truncate text
 */
function truncate(text, maxLength) {
    if (text.length <= maxLength) return text;
    return text.substr(0, maxLength) + '...';
}

/**
 * Escape HTML
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Sanitize user input to prevent XSS
 */
function sanitizeInput(input) {
    // Remove any HTML tags
    const div = document.createElement('div');
    div.textContent = input;
    return div.innerHTML;
}

/**
 * Handle invalid form submission
 */
function handleInvalidForm(event) {
    event.preventDefault();
    const field = event.target;
    
    if (field.validity.valueMissing) {
        showError('Please enter a query before searching');
    } else if (field.validity.tooLong) {
        showError(`Query is too long. Maximum ${field.maxLength} characters allowed.`);
    }
    
    field.focus();
}

/**
 * Setup keyboard navigation
 */
function setupKeyboardNavigation() {
    // Tab trap for modals (if implemented)
    const focusableElements = 'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])';
    
    // Ctrl+K or Cmd+K to focus search
    document.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            elements.queryInput.focus();
            elements.queryInput.select();
        }
    });
    
    // Arrow key navigation for history items
    elements.historyList.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
            e.preventDefault();
            const items = elements.historyList.querySelectorAll('.history-item');
            const currentIndex = Array.from(items).indexOf(document.activeElement);
            
            let nextIndex;
            if (e.key === 'ArrowDown') {
                nextIndex = currentIndex + 1 < items.length ? currentIndex + 1 : 0;
            } else {
                nextIndex = currentIndex - 1 >= 0 ? currentIndex - 1 : items.length - 1;
            }
            
            items[nextIndex]?.focus();
        }
    });
}

/**
 * Check for API key requirement
 */
async function checkAPIKeyRequirement() {
    const apiKey = localStorage.getItem('legal_retriever_api_key');
    if (apiKey) {
        // Add to all future requests
        window.API_KEY = apiKey;
    }
}

// Periodic health check
setInterval(checkAPIHealth, 30000); // Every 30 seconds

// Check API key on load
checkAPIKeyRequirement();