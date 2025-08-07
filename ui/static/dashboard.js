/**
 * Sophie HITL Dashboard - Optimized JavaScript for Modular Architecture
 * 
 * Features:
 * - Modern ES6+ syntax with async/await
 * - Improved error handling and retry logic
 * - Real-time updates with WebSocket support
 * - Better accessibility and keyboard navigation
 * - Performance optimizations with debouncing
 * - Modular component architecture
 */

class SophieDashboard {
    constructor() {
        this.apiBaseUrl = window.location.origin;
        this.refreshInterval = 30000; // 30 seconds
        this.retryAttempts = 3;
        this.retryDelay = 1000; // 1 second
        this.debounceDelay = 300;
        this.isConnected = true;
        this.lastUpdate = null;
        this.updateQueue = [];
        this.websocket = null;
        
        // Performance tracking
        this.metrics = {
            apiCalls: 0,
            errors: 0,
            lastRefresh: null,
            averageResponseTime: 0
        };
        
        this.init();
    }

    async init() {
        try {
            this.setupEventListeners();
            this.setupWebSocket();
            this.startAutoRefresh();
            await this.loadInitialData();
            this.setupPerformanceMonitoring();
        } catch (error) {
            console.error('Dashboard initialization failed:', error);
            this.showAlert('Dashboard initialization failed', 'error');
        }
    }

    setupEventListeners() {
        // Event delegation for better performance
        document.addEventListener('click', this.handleClick.bind(this));
        document.addEventListener('keydown', this.handleKeyboard.bind(this));
        
        // Setup manual refresh with debouncing
        const refreshBtn = document.getElementById('refresh-btn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', this.debounce(() => this.refreshData(), this.debounceDelay));
        }

        // Setup cleanup button
        const cleanupBtn = document.getElementById('cleanup-btn');
        if (cleanupBtn) {
            cleanupBtn.addEventListener('click', () => this.cleanupOldPlans());
        }

        // Setup search functionality
        const searchInput = document.getElementById('search-input');
        if (searchInput) {
            searchInput.addEventListener('input', this.debounce(() => this.handleSearch(), 500));
        }

        // Setup filters
        const filterSelects = document.querySelectorAll('.filter-select');
        filterSelects.forEach(select => {
            select.addEventListener('change', () => this.handleFilter());
        });

        // Setup keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                switch (e.key) {
                    case 'r':
                        e.preventDefault();
                        this.refreshData();
                        break;
                    case 'f':
                        e.preventDefault();
                        document.getElementById('search-input')?.focus();
                        break;
                }
            }
        });
    }

    setupWebSocket() {
        try {
            const wsUrl = this.apiBaseUrl.replace('http', 'ws') + '/ws';
            this.websocket = new WebSocket(wsUrl);
            
            this.websocket.onopen = () => {
                console.log('WebSocket connected');
                this.isConnected = true;
                this.showConnectionStatus('connected');
            };
            
            this.websocket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleWebSocketMessage(data);
            };
            
            this.websocket.onclose = () => {
                console.log('WebSocket disconnected');
                this.isConnected = false;
                this.showConnectionStatus('disconnected');
                // Attempt to reconnect
                setTimeout(() => this.setupWebSocket(), 5000);
            };
            
            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.isConnected = false;
                this.showConnectionStatus('error');
            };
        } catch (error) {
            console.warn('WebSocket not available, falling back to polling');
        }
    }

    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'plan_submitted':
                this.handleNewPlan(data.plan);
                break;
            case 'plan_updated':
                this.handlePlanUpdate(data.plan);
                break;
            case 'plan_decided':
                this.handlePlanDecision(data.decision);
                break;
            case 'stats_updated':
                this.updateStats(data.stats);
                break;
        }
    }

    handleClick(e) {
        const target = e.target;
        
        if (target.matches('.btn-approve')) {
            e.preventDefault();
            this.approvePlan(target.dataset.planId);
        } else if (target.matches('.btn-reject')) {
            e.preventDefault();
            this.rejectPlan(target.dataset.planId);
        } else if (target.matches('.btn-fork')) {
            e.preventDefault();
            this.forkPlan(target.dataset.planId);
        } else if (target.matches('.plan-card')) {
            this.expandPlanCard(target.closest('.plan-card'));
        } else if (target.matches('.decision-item')) {
            this.showDecisionDetails(target.closest('.decision-item'));
        }
    }

    handleKeyboard(e) {
        const target = e.target;
        
        if (target.matches('.btn') && (e.key === 'Enter' || e.key === ' ')) {
            e.preventDefault();
            target.click();
        }
        
        // Escape key to close modals
        if (e.key === 'Escape') {
            this.closeModals();
        }
    }

    debounce(func, delay) {
        let timeoutId;
        return function (...args) {
            clearTimeout(timeoutId);
            timeoutId = setTimeout(() => func.apply(this, args), delay);
        };
    }

    startAutoRefresh() {
        setInterval(() => {
            if (this.isConnected && !this.isUserActive()) {
                this.refreshData();
            }
        }, this.refreshInterval);
    }

    isUserActive() {
        // Check if user is actively interacting with the page
        return document.hasFocus() && 
               (document.querySelector(':hover') || 
                document.activeElement?.matches('input, textarea, select'));
    }

    async loadInitialData() {
        try {
            this.showLoading();
            await this.refreshData();
            this.hideLoading();
        } catch (error) {
            console.error('Failed to load initial data:', error);
            this.showAlert('Failed to load initial data', 'error');
            this.hideLoading();
        }
    }

    async refreshData() {
        const startTime = performance.now();
        
        try {
            const [pendingPlans, decisions, stats] = await Promise.allSettled([
                this.fetchPendingPlans(),
                this.fetchDecisions(),
                this.fetchStats()
            ]);

            // Update response time metric
            const responseTime = performance.now() - startTime;
            this.updateResponseTime(responseTime);

            // Handle successful responses
            if (pendingPlans.status === 'fulfilled') {
                this.updatePendingPlans(pendingPlans.value);
            } else {
                console.error('Failed to fetch pending plans:', pendingPlans.reason);
            }

            if (decisions.status === 'fulfilled') {
                this.updateDecisions(decisions.value);
            } else {
                console.error('Failed to fetch decisions:', decisions.reason);
            }

            if (stats.status === 'fulfilled') {
                this.updateStats(stats.value);
            } else {
                console.error('Failed to fetch stats:', stats.reason);
            }

            this.metrics.lastRefresh = new Date();
            this.updateLastRefreshDisplay();

        } catch (error) {
            console.error('Failed to refresh data:', error);
            this.showAlert('Failed to refresh data', 'error');
            this.metrics.errors++;
        }
    }

    async fetchWithRetry(url, options = {}) {
        let lastError;
        
        for (let attempt = 1; attempt <= this.retryAttempts; attempt++) {
            try {
                const startTime = performance.now();
                const response = await fetch(url, {
                    ...options,
                    headers: {
                        'Content-Type': 'application/json',
                        ...options.headers
                    }
                });
                
                const responseTime = performance.now() - startTime;
                this.updateResponseTime(responseTime);
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                this.metrics.apiCalls++;
                return await response.json();
                
            } catch (error) {
                lastError = error;
                console.warn(`API call attempt ${attempt} failed:`, error);
                
                if (attempt < this.retryAttempts) {
                    await this.sleep(this.retryDelay * attempt);
                }
            }
        }
        
        throw lastError;
    }

    async fetchPendingPlans() {
        return this.fetchWithRetry(`${this.apiBaseUrl}/api/plans/pending`);
    }

    async fetchDecisions() {
        return this.fetchWithRetry(`${this.apiBaseUrl}/api/decisions`);
    }

    async fetchStats() {
        return this.fetchWithRetry(`${this.apiBaseUrl}/api/stats`);
    }

    updatePendingPlans(plans) {
        const container = document.getElementById('pending-plans-container');
        if (!container) return;

        if (!Array.isArray(plans) || plans.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <h3>No pending plans</h3>
                    <p>All plans have been processed or no plans require review.</p>
                </div>
            `;
            return;
        }

        // Sort plans by priority (trust score, confidence, creation time)
        const sortedPlans = plans.sort((a, b) => {
            const priorityA = (a.trust_score * 0.4) + (a.confidence_score * 0.4) + (a.evaluation_score || 0 * 0.2);
            const priorityB = (b.trust_score * 0.4) + (b.confidence_score * 0.4) + (b.evaluation_score || 0 * 0.2);
            return priorityB - priorityA;
        });

        container.innerHTML = sortedPlans.map(plan => this.createPlanCard(plan)).join('');
        
        // Add animation classes
        const planCards = container.querySelectorAll('.plan-card');
        planCards.forEach((card, index) => {
            card.style.animationDelay = `${index * 0.1}s`;
        });
    }

    createPlanCard(plan) {
        const trustClass = this.getScoreClass(plan.trust_score);
        const confidenceClass = this.getScoreClass(plan.confidence_score);
        const evalClass = plan.evaluation_score ? this.getScoreClass(plan.evaluation_score) : '';
        const priority = this.calculatePriority(plan);

        return `
            <div class="plan-card" data-plan-id="${plan.plan_id}" data-priority="${priority}">
                <div class="plan-header">
                    <div class="plan-title">
                        <span class="priority-indicator priority-${priority}"></span>
                        Plan: ${this.escapeHtml(plan.plan_id)}
                    </div>
                    <div class="plan-scores">
                        <div class="score ${trustClass}" title="Trust Score: ${plan.trust_score.toFixed(2)}">
                            Trust: ${plan.trust_score.toFixed(2)}
                        </div>
                        <div class="score ${confidenceClass}" title="Confidence Score: ${plan.confidence_score.toFixed(2)}">
                            Confidence: ${plan.confidence_score.toFixed(2)}
                        </div>
                        ${plan.evaluation_score ? `
                            <div class="score ${evalClass}" title="Evaluation Score: ${plan.evaluation_score.toFixed(2)}">
                                Eval: ${plan.evaluation_score.toFixed(2)}
                            </div>
                        ` : ''}
                    </div>
                </div>
                <div class="plan-content">
                    <strong>Task:</strong> ${this.escapeHtml(plan.task_id)}<br>
                    <strong>Agent:</strong> ${this.escapeHtml(plan.agent_id)}<br>
                    <strong>Content:</strong><br>
                    <div class="plan-content-text" data-content="${this.escapeHtml(plan.plan_content)}">
                        ${this.truncateContent(plan.plan_content, 200)}
                    </div>
                    ${plan.plan_content.length > 200 ? `
                        <button class="btn btn-expand" onclick="this.parentElement.querySelector('.plan-content-text').textContent = this.parentElement.querySelector('.plan-content-text').dataset.content; this.remove();">
                            Show More
                        </button>
                    ` : ''}
                </div>
                <div class="plan-actions">
                    <button class="btn btn-approve" data-plan-id="${plan.plan_id}" aria-label="Approve plan ${plan.plan_id}">
                        ✅ Approve
                    </button>
                    <button class="btn btn-reject" data-plan-id="${plan.plan_id}" aria-label="Reject plan ${plan.plan_id}">
                        ❌ Reject
                    </button>
                    <button class="btn btn-fork" data-plan-id="${plan.plan_id}" aria-label="Fork plan ${plan.plan_id}">
                        ♻️ Fork
                    </button>
                </div>
            </div>
        `;
    }

    updateDecisions(decisions) {
        const container = document.getElementById('decisions-container');
        if (!container) return;

        if (!Array.isArray(decisions) || decisions.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <h3>No decisions yet</h3>
                    <p>No human decisions have been recorded.</p>
                </div>
            `;
            return;
        }

        // Sort decisions by timestamp (newest first)
        const sortedDecisions = decisions.sort((a, b) => 
            new Date(b.decision_timestamp) - new Date(a.decision_timestamp)
        );

        container.innerHTML = sortedDecisions.map(decision => this.createDecisionItem(decision)).join('');
    }

    createDecisionItem(decision) {
        const statusClass = decision.approved ? 'approved' : 'rejected';
        const statusText = decision.approved ? 'Approved' : 'Rejected';
        const timestamp = new Date(decision.decision_timestamp).toLocaleString();
        const timeAgo = this.getTimeAgo(new Date(decision.decision_timestamp));

        return `
            <div class="decision-item" data-decision-id="${decision.plan_id}">
                <div class="decision-info">
                    <div class="decision-title">${this.escapeHtml(decision.plan_id)}</div>
                    <div class="decision-meta">
                        ${timestamp} (${timeAgo})
                        ${decision.user_id ? `by ${this.escapeHtml(decision.user_id)}` : ''}
                    </div>
                    ${decision.reason ? `<div class="decision-reason">${this.escapeHtml(decision.reason)}</div>` : ''}
                </div>
                <div class="decision-status ${statusClass}">
                    ${statusText}
                </div>
            </div>
        `;
    }

    updateStats(stats) {
        const elements = {
            'pending-plans-count': stats.pending_plans_count,
            'total-decisions': stats.total_decisions,
            'recent-approvals': stats.recent_approvals,
            'recent-rejections': stats.recent_rejections
        };

        Object.entries(elements).forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element) {
                // Animate the number change
                const currentValue = parseInt(element.textContent) || 0;
                this.animateNumber(element, currentValue, value);
            }
        });

        // Update additional stats if available
        if (stats.average_response_time) {
            const responseTimeElement = document.getElementById('avg-response-time');
            if (responseTimeElement) {
                responseTimeElement.textContent = `${stats.average_response_time.toFixed(2)}ms`;
            }
        }
    }

    animateNumber(element, from, to) {
        const duration = 500;
        const start = performance.now();
        
        const animate = (currentTime) => {
            const elapsed = currentTime - start;
            const progress = Math.min(elapsed / duration, 1);
            
            const current = Math.floor(from + (to - from) * progress);
            element.textContent = current;
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };
        
        requestAnimationFrame(animate);
    }

    async approvePlan(planId) {
        try {
            const reason = await this.showPrompt('Reason for approval (optional):') || '';
            
            const response = await this.fetchWithRetry(`${this.apiBaseUrl}/api/plans/${planId}/approve`, {
                method: 'POST',
                body: JSON.stringify({ reason })
            });

            this.showAlert('Plan approved successfully!', 'success');
            await this.refreshData();
            
        } catch (error) {
            console.error('Error approving plan:', error);
            this.showAlert('Error approving plan: ' + error.message, 'error');
        }
    }

    async rejectPlan(planId) {
        try {
            const reason = await this.showPrompt('Reason for rejection:');
            if (reason === null) return; // User cancelled

            const response = await this.fetchWithRetry(`${this.apiBaseUrl}/api/plans/${planId}/reject`, {
                method: 'POST',
                body: JSON.stringify({ reason })
            });

            this.showAlert('Plan rejected successfully!', 'success');
            await this.refreshData();
            
        } catch (error) {
            console.error('Error rejecting plan:', error);
            this.showAlert('Error rejecting plan: ' + error.message, 'error');
        }
    }

    async forkPlan(planId) {
        try {
            const response = await this.fetchWithRetry(`${this.apiBaseUrl}/api/plans/${planId}/fork`, {
                method: 'POST',
                body: JSON.stringify({ 
                    reason: 'Plan forked for regeneration',
                    modifications: {}
                })
            });

            this.showAlert('Plan forked successfully!', 'success');
            await this.refreshData();
            
        } catch (error) {
            console.error('Error forking plan:', error);
            this.showAlert('Error forking plan: ' + error.message, 'error');
        }
    }

    async cleanupOldPlans() {
        try {
            const maxAge = await this.showPrompt('Maximum age of plans to keep (in hours):', '24');
            if (maxAge === null) return;

            const response = await this.fetchWithRetry(`${this.apiBaseUrl}/api/plans/cleanup`, {
                method: 'POST',
                body: JSON.stringify({ max_age_hours: parseInt(maxAge) })
            });

            this.showAlert(`Cleaned up ${response.removed_plans.length} old plans`, 'success');
            await this.refreshData();
            
        } catch (error) {
            console.error('Error cleaning up plans:', error);
            this.showAlert('Error cleaning up plans: ' + error.message, 'error');
        }
    }

    showAlert(message, type = 'info') {
        // Remove existing alerts
        const existingAlerts = document.querySelectorAll('.alert');
        existingAlerts.forEach(alert => alert.remove());

        // Create new alert
        const alert = document.createElement('div');
        alert.className = `alert alert-${type}`;
        alert.innerHTML = `
            <span>${this.escapeHtml(message)}</span>
            <button class="alert-close" onclick="this.parentElement.remove()">&times;</button>
        `;

        // Insert at top of container
        const container = document.querySelector('.container');
        if (container) {
            container.insertBefore(alert, container.firstChild);
        }

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (alert.parentNode) {
                alert.parentNode.removeChild(alert);
            }
        }, 5000);
    }

    async showPrompt(message, defaultValue = '') {
        return new Promise((resolve) => {
            const result = prompt(message, defaultValue);
            resolve(result);
        });
    }

    getScoreClass(score) {
        if (score >= 0.7) return 'high';
        if (score >= 0.4) return 'medium';
        return 'low';
    }

    calculatePriority(plan) {
        const trustWeight = 0.4;
        const confidenceWeight = 0.4;
        const evalWeight = 0.2;
        
        const priority = (plan.trust_score * trustWeight) + 
                       (plan.confidence_score * confidenceWeight) + 
                       ((plan.evaluation_score || 0) * evalWeight);
        
        if (priority >= 0.7) return 'high';
        if (priority >= 0.4) return 'medium';
        return 'low';
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    truncateContent(text, maxLength) {
        if (text.length <= maxLength) return this.escapeHtml(text);
        return this.escapeHtml(text.substring(0, maxLength)) + '...';
    }

    getTimeAgo(date) {
        const now = new Date();
        const diff = now - date;
        const minutes = Math.floor(diff / 60000);
        const hours = Math.floor(diff / 3600000);
        const days = Math.floor(diff / 86400000);

        if (days > 0) return `${days}d ago`;
        if (hours > 0) return `${hours}h ago`;
        if (minutes > 0) return `${minutes}m ago`;
        return 'Just now';
    }

    updateResponseTime(responseTime) {
        this.metrics.averageResponseTime = 
            (this.metrics.averageResponseTime + responseTime) / 2;
    }

    updateLastRefreshDisplay() {
        const element = document.getElementById('last-refresh');
        if (element && this.metrics.lastRefresh) {
            element.textContent = this.metrics.lastRefresh.toLocaleTimeString();
        }
    }

    showConnectionStatus(status) {
        const element = document.getElementById('connection-status');
        if (element) {
            element.className = `status-indicator ${status}`;
            element.title = `Connection: ${status}`;
        }
    }

    showLoading() {
        const loadingElement = document.getElementById('loading-overlay');
        if (loadingElement) {
            loadingElement.style.display = 'flex';
        }
    }

    hideLoading() {
        const loadingElement = document.getElementById('loading-overlay');
        if (loadingElement) {
            loadingElement.style.display = 'none';
        }
    }

    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    setupPerformanceMonitoring() {
        // Monitor memory usage
        if ('memory' in performance) {
            setInterval(() => {
                const memory = performance.memory;
                if (memory.usedJSHeapSize > memory.jsHeapSizeLimit * 0.8) {
                    console.warn('High memory usage detected');
                }
            }, 30000);
        }

        // Monitor long tasks
        if ('PerformanceObserver' in window) {
            const observer = new PerformanceObserver((list) => {
                for (const entry of list.getEntries()) {
                    if (entry.duration > 50) {
                        console.warn('Long task detected:', entry);
                    }
                }
            });
            observer.observe({ entryTypes: ['longtask'] });
        }
    }

    handleSearch() {
        const searchTerm = document.getElementById('search-input')?.value.toLowerCase();
        const planCards = document.querySelectorAll('.plan-card');
        
        planCards.forEach(card => {
            const content = card.textContent.toLowerCase();
            const matches = content.includes(searchTerm);
            card.style.display = matches ? 'block' : 'none';
        });
    }

    handleFilter() {
        // Implement filtering logic based on selected filters
        const filters = document.querySelectorAll('.filter-select');
        const activeFilters = Array.from(filters)
            .filter(select => select.value !== 'all')
            .map(select => ({ type: select.name, value: select.value }));
        
        // Apply filters to plan cards
        const planCards = document.querySelectorAll('.plan-card');
        planCards.forEach(card => {
            const shouldShow = this.matchesFilters(card, activeFilters);
            card.style.display = shouldShow ? 'block' : 'none';
        });
    }

    matchesFilters(card, filters) {
        return filters.every(filter => {
            const cardValue = card.dataset[filter.type];
            return cardValue === filter.value;
        });
    }

    expandPlanCard(card) {
        const content = card.querySelector('.plan-content-text');
        const expandBtn = card.querySelector('.btn-expand');
        
        if (content && expandBtn) {
            content.textContent = content.dataset.content;
            expandBtn.remove();
        }
    }

    showDecisionDetails(decisionItem) {
        // Implement modal or expandable details view
        console.log('Show decision details:', decisionItem.dataset.decisionId);
    }

    closeModals() {
        // Close any open modals or expanded views
        const modals = document.querySelectorAll('.modal');
        modals.forEach(modal => modal.style.display = 'none');
    }
}

// Utility functions
function formatTimestamp(timestamp) {
    return new Date(timestamp).toLocaleString();
}

function formatScore(score) {
    return (score * 100).toFixed(1) + '%';
}

function showLoading(element) {
    element.innerHTML = '<div class="loading"></div> Loading...';
}

function hideLoading(element, originalContent) {
    element.innerHTML = originalContent;
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.sophieDashboard = new SophieDashboard();
});

// Export for use in other scripts
window.SophieDashboard = SophieDashboard; 