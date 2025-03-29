document.addEventListener('DOMContentLoaded', function() {
    // Initialize components
    initializeSelectors();
    fetchOptionChain();
    
    // Set up refresh interval
    setInterval(fetchOptionChain, 30000); // Update every 30 seconds
    
    // Set up event listeners
    document.getElementById('refreshDataBtn').addEventListener('click', fetchOptionChain);
    document.getElementById('indexSelect').addEventListener('change', function() {
        setIndex(this.value);
        fetchExpiryDates(this.value);
    });
    document.getElementById('expirySelect').addEventListener('change', function() {
        setExpiry(this.value);
        fetchOptionChain();
    });
    
    // Auto trading switch
    document.getElementById('autoTradeSwitch').addEventListener('change', function() {
        toggleAutoTrading(this.checked);
    });
    
    // Initialize charts
    initializeCharts();
});

// Initialize charts
function initializeCharts() {
    const positionChartCtx = document.getElementById('positionChart').getContext('2d');
    window.positionChart = new Chart(positionChartCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Portfolio Value',
                data: [],
                borderColor: 'rgba(75, 192, 192, 1)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                tension: 0.1,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Time'
                    }
                },
                y: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Value (₹)'
                    }
                }
            }
        }
    });
}

// Initialize selectors
function initializeSelectors() {
    // Populate initial expiry dates
    fetchExpiryDates(document.getElementById('indexSelect').value);
}

// Fetch option chain data
function fetchOptionChain() {
    const index = document.getElementById('indexSelect').value;
    const expiry = document.getElementById('expirySelect').value;
    
    fetch(`/api/option_chain?index=${index}&expiry=${expiry}`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                updateOptionChain(data);
                updateLastUpdated(data.timestamp);
                
                // Update live market parameters display
                updateLiveMarketDisplay(data);
            } else {
                console.error('Error fetching option chain:', data.error);
            }
        })
        .catch(error => console.error('Error:', error));
}

// Update live market parameters display
function updateLiveMarketDisplay(data) {
    // Update market data in the live trading section
    if (data.spot_price) {
        document.getElementById('spotPriceValue').textContent = `₹${data.spot_price.toLocaleString('en-IN', {maximumFractionDigits: 2})}`;
    }
    
    // Calculate PCR (Put-Call Ratio) from the data
    if (data.calls && data.puts) {
        const callOI = data.calls.reduce((sum, call) => sum + call.open_interest, 0);
        const putOI = data.puts.reduce((sum, put) => sum + put.open_interest, 0);
        const pcr = putOI / (callOI || 1);  // Avoid division by zero
        document.getElementById('pcrValue').textContent = pcr.toFixed(2);
        
        // Color-code PCR (>1 is bullish bias, <1 is bearish bias)
        const pcrElement = document.getElementById('pcrValue');
        if (pcr > 1.2) {
            pcrElement.className = 'badge bg-success';
        } else if (pcr < 0.8) {
            pcrElement.className = 'badge bg-danger';
        } else {
            pcrElement.className = 'badge bg-warning';
        }
    }
    
    // Calculate IV percentile if we have enough data
    if (data.calls && data.calls.length > 0) {
        // Find ATM call
        const atmCall = data.calls.reduce((closest, call) => {
            return Math.abs(call.strike_price - data.spot_price) < Math.abs(closest.strike_price - data.spot_price) ? call : closest;
        }, data.calls[0]);
        
        // Update IV display
        if (atmCall && atmCall.implied_volatility) {
            document.getElementById('ivValue').textContent = `${(atmCall.implied_volatility * 100).toFixed(1)}%`;
        }
    }
}

// Fetch available expiry dates
function fetchExpiryDates(index) {
    fetch(`/api/available_expiries?index=${index}`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                updateExpirySelect(data.expiries);
            } else {
                console.error('Error fetching expiry dates:', data.error);
            }
        })
        .catch(error => console.error('Error:', error));
}

// Set index
function setIndex(index) {
    fetch('/api/set_index', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            index: index
        })
    })
    .then(response => response.json())
    .then(data => {
        if (!data.success) {
            console.error('Error setting index:', data.error);
        }
    })
    .catch(error => console.error('Error:', error));
}

// Set expiry
function setExpiry(expiry) {
    fetch('/api/set_expiry', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            expiry: expiry
        })
    })
    .then(response => response.json())
    .then(data => {
        if (!data.success) {
            console.error('Error setting expiry:', data.error);
        }
    })
    .catch(error => console.error('Error:', error));
}

// Toggle auto trading
function toggleAutoTrading(enabled) {
    console.log(`Auto trading ${enabled ? 'enabled' : 'disabled'}`);
    // In a real implementation, this would communicate with the backend
    // to start or stop automatic trading
    const tradingModeSelect = document.getElementById('tradingMode');
    const tradingMode = tradingModeSelect.value;
    
    // Update the model confidence based on trading mode
    const confidence = Math.random() * 40 + 60; // Random between 60-100%
    document.getElementById('modelConfidence').style.width = `${confidence}%`;
    document.getElementById('modelConfidence').textContent = `${Math.round(confidence)}%`;
    
    // Show notification
    if (enabled) {
        // In a real implementation, this would be based on model prediction quality
        if (confidence > 80) {
            showNotification('Model confidence is high, favorable conditions for trading', 'success');
        } else {
            showNotification('Auto trading enabled with moderate confidence', 'warning');
        }
    }
}

// Show notification
function showNotification(message, type = 'info') {
    // Add toast container if it doesn't exist
    if (!document.getElementById('toastContainer')) {
        const container = document.createElement('div');
        container.className = 'toast-container position-fixed bottom-0 end-0 p-3';
        container.id = 'toastContainer';
        document.body.appendChild(container);
    }
    
    // Create toast
    const id = 'toast' + Date.now();
    const toast = document.createElement('div');
    toast.className = `toast show bg-${type}`;
    toast.id = id;
    toast.setAttribute('role', 'alert');
    toast.setAttribute('aria-live', 'assertive');
    toast.setAttribute('aria-atomic', 'true');
    
    toast.innerHTML = `
        <div class="toast-header">
            <strong class="me-auto">RL Options Trader</strong>
            <small>Just now</small>
            <button type="button" class="btn-close" data-bs-dismiss="toast" aria-label="Close"></button>
        </div>
        <div class="toast-body">
            ${message}
        </div>
    `;
    
    document.getElementById('toastContainer').appendChild(toast);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        toast.remove();
    }, 5000);
}

// Update option chain display
function updateOptionChain(data) {
    const tbody = document.getElementById('optionChainBody');
    tbody.innerHTML = '';
    
    if (!data.calls || !data.puts || data.calls.length === 0 || data.puts.length === 0) {
        tbody.innerHTML = '<tr><td colspan="13" class="text-center">No option chain data available</td></tr>';
        return;
    }
    
    // Get all strikes
    const allStrikes = [...new Set([...data.calls.map(call => call.strike_price), ...data.puts.map(put => put.strike_price)])].sort((a, b) => a - b);
    
    // Find ATM strike
    const spotPrice = data.spot_price;
    const atmStrike = allStrikes.reduce((prev, curr) => {
        return (Math.abs(curr - spotPrice) < Math.abs(prev - spotPrice) ? curr : prev);
    });
    
    // Create a map for quick lookup
    const callMap = {};
    data.calls.forEach(call => {
        callMap[call.strike_price] = call;
    });
    
    const putMap = {};
    data.puts.forEach(put => {
        putMap[put.strike_price] = put;
    });
    
    // Generate rows
    allStrikes.forEach(strike => {
        const call = callMap[strike];
        const put = putMap[strike];
        
        const tr = document.createElement('tr');
        
        // Highlight ATM strike
        if (strike === atmStrike) {
            tr.className = 'table-active';
        }
        
        // Call data
        if (call) {
            tr.innerHTML += `
                <td>${formatOI(call.open_interest)}</td>
                <td>${formatVolume(call.volume)}</td>
                <td>${formatIV(call.implied_volatility)}</td>
                <td>${formatPrice(call.last_price)}</td>
                <td class="${call.change >= 0 ? 'text-success' : 'text-danger'}">${formatChange(call.change_percentage)}</td>
                <td>${formatDelta(call.delta)}</td>
            `;
        } else {
            tr.innerHTML += '<td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td>';
        }
        
        // Strike
        tr.innerHTML += `<td class="text-center fw-bold">${strike}</td>`;
        
        // Put data
        if (put) {
            tr.innerHTML += `
                <td>${formatDelta(put.delta)}</td>
                <td class="${put.change >= 0 ? 'text-success' : 'text-danger'}">${formatChange(put.change_percentage)}</td>
                <td>${formatPrice(put.last_price)}</td>
                <td>${formatIV(put.implied_volatility)}</td>
                <td>${formatVolume(put.volume)}</td>
                <td>${formatOI(put.open_interest)}</td>
            `;
        } else {
            tr.innerHTML += '<td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td>';
        }
        
        tbody.appendChild(tr);
    });
    
    // Simulate position updates (in a real implementation this would be based on actual positions)
    simulatePositionUpdates();
}

// Format functions
function formatOI(oi) {
    return oi ? oi.toLocaleString('en-IN') : '-';
}

function formatVolume(vol) {
    return vol ? vol.toLocaleString('en-IN') : '-';
}

function formatIV(iv) {
    return iv ? `${(iv * 100).toFixed(1)}%` : '-';
}

function formatPrice(price) {
    return price ? `₹${price.toFixed(2)}` : '-';
}

function formatChange(change) {
    return change ? `${change >= 0 ? '+' : ''}${change.toFixed(2)}%` : '-';
}

function formatDelta(delta) {
    return delta ? delta.toFixed(2) : '-';
}

// Update last updated timestamp
function updateLastUpdated(timestamp) {
    const element = document.getElementById('lastUpdated');
    if (element) {
        const date = new Date(timestamp);
        element.textContent = `Last updated: ${date.toLocaleTimeString()}`;
    }
}

// Update expiry select dropdown
function updateExpirySelect(expiries) {
    const select = document.getElementById('expirySelect');
    const currentValue = select.value;
    
    // Clear current options
    select.innerHTML = '';
    
    // Add new options
    expiries.forEach(expiry => {
        const option = document.createElement('option');
        option.value = expiry;
        option.textContent = formatExpiryDate(expiry);
        select.appendChild(option);
    });
    
    // Try to restore previous selection
    if (currentValue && expiries.includes(currentValue)) {
        select.value = currentValue;
    } else if (expiries.length > 0) {
        // Set to first expiry and notify backend
        setExpiry(expiries[0]);
    }
}

// Format expiry date for display
function formatExpiryDate(dateStr) {
    try {
        const date = new Date(dateStr);
        return date.toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' });
    } catch (e) {
        return dateStr;
    }
}

// Simulate position updates (for demonstration)
function simulatePositionUpdates() {
    // Update current P&L
    const pnl = (Math.random() * 2000 - 1000).toFixed(2);
    const pnlElement = document.getElementById('currentPnL');
    pnlElement.textContent = `₹${pnl}`;
    pnlElement.className = parseFloat(pnl) >= 0 ? 'text-success' : 'text-danger';
    
    // Update portfolio chart
    const chart = window.positionChart;
    if (chart) {
        // Add a new data point
        const now = new Date();
        const timeLabel = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        
        // Add to chart if we don't have too many points
        if (chart.data.labels.length > 20) {
            chart.data.labels.shift();
            chart.data.datasets[0].data.shift();
        }
        
        // Get previous value or set initial value
        let prevValue = 100000;
        if (chart.data.datasets[0].data.length > 0) {
            prevValue = chart.data.datasets[0].data[chart.data.datasets[0].data.length - 1];
        }
        
        // Add small random change to previous value
        const newValue = prevValue + (Math.random() * 1000 - 500);
        
        chart.data.labels.push(timeLabel);
        chart.data.datasets[0].data.push(newValue);
        chart.update();
    }
}