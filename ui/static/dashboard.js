// Sophie HITL Dashboard JavaScript

class SophieDashboard {
    constructor() {
        this.apiBaseUrl = window.location.origin;
        this.refreshInterval = 30000; // 30 seconds
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.startAutoRefresh();
        this.loadInitialData();
    }

    setupEventListeners() {
        // Setup plan action buttons
        document.addEventListener('click', (e) => {
            if (e.target.matches('.btn-approve')) {
                e.preventDefault();
                this.approvePlan(e.target.dataset.planId);
            } else if (e.target.matches('.btn-reject')) {
                e.preventDefault();
                this.rejectPlan(e.target.dataset.planId);
            } else if (e.target.matches('.btn-fork')) {
                e.preventDefault();
                this.forkPlan(e.target.dataset.planId);
            }
        });

        // Setup manual refresh
        const refreshBtn = document.getElementById('refresh-btn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.refreshData());
        }

        // Setup cleanup button
        const cleanupBtn = document.getElementById('cleanup-btn');
        if (cleanupBtn) {
            cleanupBtn.addEventListener('click', () => this.cleanupOldPlans());
        }
    }

    startAutoRefresh() {
        setInterval(() => {
            this.refreshData();
        }, this.refreshInterval);
    }

    async loadInitialData() {
        try {
            await this.refreshData();
        } catch (error) {
            console.error('Failed to load initial data:', error);
            this.showAlert('Failed to load initial data', 'error');
        }
    }

    async refreshData() {
        try {
            const [pendingPlans, decisions, stats] = await Promise.all([
                this.fetchPendingPlans(),
                this.fetchDecisions(),
                this.fetchStats()
            ]);

            this.updatePendingPlans(pendingPlans);
            this.updateDecisions(decisions);
            this.updateStats(stats);
        } catch (error) {
            console.error('Failed to refresh data:', error);
            this.showAlert('Failed to refresh data', 'error');
        }
    }

    async fetchPendingPlans() {
        const response = await fetch(`${this.apiBaseUrl}/api/plans/pending`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        return data.plans;
    }

    async fetchDecisions() {
        const response = await fetch(`${this.apiBaseUrl}/api/decisions`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        return data.decisions;
    }

    async fetchStats() {
        const response = await fetch(`${this.apiBaseUrl}/api/stats`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return await response.json();
    }

    updatePendingPlans(plans) {
        const container = document.getElementById('pending-plans-container');
        if (!container) return;

        if (plans.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <h3>No pending plans</h3>
                    <p>All plans have been processed or no plans require review.</p>
                </div>
            `;
            return;
        }

        container.innerHTML = plans.map(plan => this.createPlanCard(plan)).join('');
    }

    createPlanCard(plan) {
        const trustClass = this.getScoreClass(plan.trust_score);
        const confidenceClass = this.getScoreClass(plan.confidence_score);
        const evalClass = plan.evaluation_score ? this.getScoreClass(plan.evaluation_score) : '';

        return `
            <div class="plan-card">
                <div class="plan-header">
                    <div class="plan-title">Plan: ${plan.plan_id}</div>
                    <div class="plan-scores">
                        <div class="score ${trustClass}">
                            Trust: ${plan.trust_score.toFixed(2)}
                        </div>
                        <div class="score ${confidenceClass}">
                            Confidence: ${plan.confidence_score.toFixed(2)}
                        </div>
                        ${plan.evaluation_score ? `
                            <div class="score ${evalClass}">
                                Eval: ${plan.evaluation_score.toFixed(2)}
                            </div>
                        ` : ''}
                    </div>
                </div>
                <div class="plan-content">
                    <strong>Task:</strong> ${plan.task_id}<br>
                    <strong>Agent:</strong> ${plan.agent_id}<br>
                    <strong>Content:</strong><br>
                    <div class="plan-content-text">${this.escapeHtml(plan.plan_content)}</div>
                </div>
                <div class="plan-actions">
                    <button class="btn btn-approve" data-plan-id="${plan.planId}">
                        ✅ Approve
                    </button>
                    <button class="btn btn-reject" data-plan-id="${plan.planId}">
                        ❌ Reject
                    </button>
                    <button class="btn btn-fork" data-plan-id="${plan.planId}">
                        ♻️ Fork
                    </button>
                </div>
            </div>
        `;
    }

    updateDecisions(decisions) {
        const container = document.getElementById('decisions-container');
        if (!container) return;

        if (decisions.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <h3>No decisions yet</h3>
                    <p>No human decisions have been recorded.</p>
                </div>
            `;
            return;
        }

        container.innerHTML = decisions.map(decision => this.createDecisionItem(decision)).join('');
    }

    createDecisionItem(decision) {
        const statusClass = decision.approved ? 'approved' : 'rejected';
        const statusText = decision.approved ? 'Approved' : 'Rejected';
        const timestamp = new Date(decision.decision_timestamp).toLocaleString();

        return `
            <div class="decision-item">
                <div class="decision-info">
                    <div class="decision-title">${decision.plan_id}</div>
                    <div class="decision-meta">
                        ${timestamp}
                        ${decision.user_id ? `by ${decision.user_id}` : ''}
                    </div>
                    ${decision.reason ? `<div class="decision-reason">${decision.reason}</div>` : ''}
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
                element.textContent = value;
            }
        });
    }

    async approvePlan(planId) {
        try {
            const reason = prompt('Reason for approval (optional):') || '';
            
            const response = await fetch(`${this.apiBaseUrl}/api/plans/${planId}/approve`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ reason })
            });

            if (response.ok) {
                this.showAlert('Plan approved successfully!', 'success');
                await this.refreshData();
            } else {
                const error = await response.json();
                this.showAlert(`Failed to approve plan: ${error.detail}`, 'error');
            }
        } catch (error) {
            console.error('Error approving plan:', error);
            this.showAlert('Error approving plan: ' + error.message, 'error');
        }
    }

    async rejectPlan(planId) {
        try {
            const reason = prompt('Reason for rejection:');
            if (reason === null) return; // User cancelled

            const response = await fetch(`${this.apiBaseUrl}/api/plans/${planId}/reject`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ reason })
            });

            if (response.ok) {
                this.showAlert('Plan rejected successfully!', 'success');
                await this.refreshData();
            } else {
                const error = await response.json();
                this.showAlert(`Failed to reject plan: ${error.detail}`, 'error');
            }
        } catch (error) {
            console.error('Error rejecting plan:', error);
            this.showAlert('Error rejecting plan: ' + error.message, 'error');
        }
    }

    async forkPlan(planId) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/plans/${planId}/fork`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    reason: 'Plan forked for regeneration',
                    modifications: {}
                })
            });

            if (response.ok) {
                this.showAlert('Plan forked successfully!', 'success');
                await this.refreshData();
            } else {
                const error = await response.json();
                this.showAlert(`Failed to fork plan: ${error.detail}`, 'error');
            }
        } catch (error) {
            console.error('Error forking plan:', error);
            this.showAlert('Error forking plan: ' + error.message, 'error');
        }
    }

    async cleanupOldPlans() {
        try {
            const maxAge = prompt('Maximum age of plans to keep (in hours):', '24');
            if (maxAge === null) return;

            const response = await fetch(`${this.apiBaseUrl}/api/plans/cleanup`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ max_age_hours: parseInt(maxAge) })
            });

            if (response.ok) {
                const result = await response.json();
                this.showAlert(`Cleaned up ${result.removed_plans.length} old plans`, 'success');
                await this.refreshData();
            } else {
                const error = await response.json();
                this.showAlert(`Failed to cleanup plans: ${error.detail}`, 'error');
            }
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
        alert.textContent = message;

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

    getScoreClass(score) {
        if (score >= 0.7) return 'high';
        if (score >= 0.4) return 'medium';
        return 'low';
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
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