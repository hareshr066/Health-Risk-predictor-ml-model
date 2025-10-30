// =======================================
// INTERACTIVE DASHBOARD LOGIC (Chart.js)
// =======================================

document.addEventListener('DOMContentLoaded', function() {
    const riskChartElement = document.getElementById('riskChart');

    if (riskChartElement) {
        fetchHistoryData(riskChartElement);
    }

    // You can add more complex JS features here later, like form validation or tracking logs.
});

function fetchHistoryData(chartElement) {
    // Show a small loading indicator while fetching data
    // document.getElementById('loadingStatus').style.display = 'block';

    fetch('/api/history')
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to fetch history data.');
            }
            return response.json();
        })
        .then(data => {
            // document.getElementById('loadingStatus').style.display = 'none';
            if (data.data.length > 0) {
                renderChart(chartElement, data.labels, data.data);
            } else {
                chartElement.parentElement.innerHTML = '<p class="text-muted" style="text-align: center;">Perform an analysis to view your history!</p>';
            }
        })
        .catch(error => {
            console.error('Error fetching history:', error);
            chartElement.parentElement.innerHTML = '<p class="error-message">Error loading chart data. Please check connection.</p>';
        });
}

function renderChart(chartElement, labels, data) {
    const ctx = chartElement.getContext('2d');

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels, // Dates/Times
            datasets: [{
                label: 'Risk Percentage (%)',
                data: data, // Risk Scores
                borderColor: 'var(--accent-color)',
                backgroundColor: 'rgba(63, 81, 181, 0.2)', // Light fill under line
                tension: 0.3,
                fill: true,
                pointRadius: 5,
                pointHoverRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false, // Allows chart to resize freely
            plugins: {
                legend: { display: false },
                tooltip: { mode: 'index', intersect: false }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: { display: true, text: 'Risk Score (%)' }
                }
            }
        }
    });
}