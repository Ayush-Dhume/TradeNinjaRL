/**
 * Chart utilities for RL Options Trading System
 * Creates and updates various charts used throughout the application
 */

/**
 * Create position monitoring chart
 * @param {CanvasRenderingContext2D} ctx - Canvas context
 * @returns {Chart} Chart instance
 */
function createPositionChart(ctx) {
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Account Equity',
                    data: [],
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.1)',
                    borderWidth: 2,
                    tension: 0.2,
                    yAxisID: 'y'
                },
                {
                    label: 'Position Value',
                    data: [],
                    borderColor: 'rgba(255, 159, 64, 1)',
                    backgroundColor: 'rgba(255, 159, 64, 0.1)',
                    borderWidth: 1,
                    tension: 0.1,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Time'
                    },
                    grid: {
                        display: false
                    }
                },
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'Equity (₹)'
                    }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Position Value (₹)'
                    },
                    grid: {
                        drawOnChartArea: false
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            if (context.parsed.y !== null) {
                                label += new Intl.NumberFormat('en-IN', { 
                                    style: 'currency', 
                                    currency: 'INR',
                                    maximumFractionDigits: 0
                                }).format(context.parsed.y);
                            }
                            return label;
                        }
                    }
                }
            }
        }
    });
}

/**
 * Create learning curve chart for training page
 * @param {CanvasRenderingContext2D} ctx - Canvas context
 * @returns {Chart} Chart instance
 */
function createLearningCurveChart(ctx) {
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Mean Reward',
                    data: [],
                    borderColor: 'rgba(54, 162, 235, 1)',
                    backgroundColor: 'rgba(54, 162, 235, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.2
                },
                {
                    label: '10-Episode Moving Average',
                    data: [],
                    borderColor: 'rgba(255, 99, 132, 1)',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Training Steps'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Reward'
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            }
        }
    });
}

/**
 * Update learning curve chart with new data
 * @param {Chart} chart - Chart.js instance
 * @param {Array} metrics - Array of training metrics
 */
function updateLearningCurveChart(chart, metrics) {
    if (!metrics || metrics.length === 0) return;
    
    // Sort metrics by steps
    metrics.sort((a, b) => a.steps - b.steps);
    
    // Extract data
    const steps = metrics.map(m => m.steps);
    const rewards = metrics.map(m => m.mean_reward);
    
    // Calculate moving average
    const movingAvg = [];
    const windowSize = 10;
    
    for (let i = 0; i < rewards.length; i++) {
        if (i < windowSize) {
            // Not enough data points yet, use average of available points
            movingAvg.push(rewards.slice(0, i + 1).reduce((a, b) => a + b, 0) / (i + 1));
        } else {
            // Use the last windowSize points
            movingAvg.push(rewards.slice(i - windowSize + 1, i + 1).reduce((a, b) => a + b, 0) / windowSize);
        }
    }
    
    // Update chart
    chart.data.labels = steps;
    chart.data.datasets[0].data = rewards;
    chart.data.datasets[1].data = movingAvg;
    chart.update();
}

/**
 * Create equity curve chart for backtesting page
 * @param {CanvasRenderingContext2D} ctx - Canvas context
 * @returns {Chart} Chart instance
 */
function createEquityCurveChart(ctx) {
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Account Equity',
                    data: [],
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.1)',
                    borderWidth: 2,
                    tension: 0.2,
                    fill: true,
                    yAxisID: 'y'
                },
                {
                    label: 'Drawdown',
                    data: [],
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    borderWidth: 1,
                    tension: 0.1,
                    fill: true,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Date'
                    }
                },
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'Equity (₹)'
                    },
                    beginAtZero: false
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    min: 0,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Drawdown (%)'
                    },
                    reverse: true,
                    grid: {
                        drawOnChartArea: false
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            if (context.parsed.y !== null) {
                                if (context.datasetIndex === 0) {
                                    label += new Intl.NumberFormat('en-IN', { 
                                        style: 'currency', 
                                        currency: 'INR',
                                        maximumFractionDigits: 0
                                    }).format(context.parsed.y);
                                } else {
                                    label += context.parsed.y.toFixed(2) + '%';
                                }
                            }
                            return label;
                        }
                    }
                }
            }
        }
    });
}

/**
 * Update equity curve chart with new data
 * @param {Chart} chart - Chart.js instance
 * @param {Object} data - Object containing labels, equity values, and drawdowns
 */
function updateEquityCurveChart(chart, data) {
    if (!data || !data.labels || !data.equity || !data.drawdowns) return;
    
    // Update chart data
    chart.data.labels = data.labels;
    chart.data.datasets[0].data = data.equity;
    chart.data.datasets[1].data = data.drawdowns;
    chart.update();
}

/**
 * Create trade distribution chart for backtesting page
 * @param {CanvasRenderingContext2D} ctx - Canvas context
 * @returns {Chart} Chart instance
 */
function createTradeDistributionChart(ctx) {
    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Winners', 'Losers'],
            datasets: [{
                label: 'Trade Outcomes',
                data: [0, 0],
                backgroundColor: [
                    'rgba(75, 192, 192, 0.7)',
                    'rgba(255, 99, 132, 0.7)'
                ],
                borderColor: [
                    'rgba(75, 192, 192, 1)',
                    'rgba(255, 99, 132, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Number of Trades'
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}

/**
 * Update trade distribution chart with new data
 * @param {Chart} chart - Chart.js instance
 * @param {Object} data - Object containing trade distribution data
 */
function updateTradeDistributionChart(chart, data) {
    if (!data || !data.labels || !data.data) return;
    
    // Update chart data
    chart.data.labels = data.labels;
    chart.data.datasets[0].data = data.data;
    chart.update();
}
