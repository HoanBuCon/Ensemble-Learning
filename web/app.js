document.addEventListener('DOMContentLoaded', () => {
    const themeToggleBtn = document.getElementById('themeToggleBtn');
    const themeIcon = document.getElementById('themeIcon');
    const themeText = document.getElementById('themeText');
    const modeOptions = document.querySelectorAll('.mode-option');
    const radioButtons = document.querySelectorAll('input[name="mode_type"]');
    const subBaseBox = document.getElementById('subBaseBox');
    const subEnsembleBox = document.getElementById('subEnsembleBox');
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const fileCount = document.getElementById('fileCount');
    const btnRun = document.getElementById('btnRun');
    const btnClear = document.getElementById('btnClear');
    const pipelineForm = document.getElementById('pipelineForm');
    const resultsContainer = document.getElementById('resultsContainer');
    const badgeStats = document.getElementById('badgeStats');

    let selectedFiles = [];

    let lastAnalyticsDataCache = null;
    let chartInstanceBenchmark = null;
    let chartInstanceLoss = null;
    let chartInstancePerClass = null;
    let chartInstanceEnsembleBenchmark = null;
    let chartInstanceEnsembleGain = null;
    let chartInstanceEnsemblePerClass = null;
    let chartInstanceBaseVsEnsemble = null;
    let chartInstanceEce = null;
    let chartInstanceBrierNll = null;

    // Theme Switcher Logic
    function applyTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('theme', theme);
        if (theme === 'light') {
            themeIcon.innerHTML = '<i class="fa-solid fa-sun"></i>';
            themeText.textContent = 'Light Mode';
        } else {
            themeIcon.innerHTML = '<i class="fa-solid fa-moon"></i>';
            themeText.textContent = 'Dark Mode';
        }

        // Re-render charts dynamically if analytics is active
        if (lastAnalyticsDataCache) {
            renderCharts(lastAnalyticsDataCache);
        }
    }

    const savedTheme = localStorage.getItem('theme') || 
        (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
    applyTheme(savedTheme);

    if (themeToggleBtn) {
        themeToggleBtn.addEventListener('click', () => {
            const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            applyTheme(newTheme);
        });
    }

    // Analytics & Benchmark Charts Logic
    const analyticsToggleBtn = document.getElementById('analyticsToggleBtn');
    const analyticsCloseBtn = document.getElementById('analyticsCloseBtn');
    const analyticsSection = document.getElementById('analyticsSection');

    let analyticsLoaded = false;
    let currentPerClassMetric = 'f1_score';
    let currentEnsPerClassMetric = 'f1_score';

    // Analytics Tab Switcher (Base Models vs Ensemble Methods)
    const analyticsTabGroup = document.getElementById('analyticsTabGroup');
    const analyticsTabBaseModels = document.getElementById('analyticsTabBaseModels');
    const analyticsTabEnsembleMethods = document.getElementById('analyticsTabEnsembleMethods');

    const analyticsTabVerification = document.getElementById('analyticsTabVerification');
    const protocolSwitcher = document.getElementById('protocolSwitcher');
    let currentProtocol = 'oof';

    if (protocolSwitcher) {
        const pPills = protocolSwitcher.querySelectorAll('.protocol-pill');
        pPills.forEach(pill => {
            pill.addEventListener('click', () => {
                pPills.forEach(p => p.classList.remove('active'));
                pill.classList.add('active');
                currentProtocol = pill.getAttribute('data-protocol');
                loadAndRenderAnalytics();
            });
        });
    }

    if (analyticsTabGroup) {
        const tabBtns = analyticsTabGroup.querySelectorAll('.analytics-tab-btn');
        tabBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                tabBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');

                const targetTab = btn.getAttribute('data-tab');
                if (analyticsTabBaseModels) analyticsTabBaseModels.style.display = (targetTab === 'base_models') ? 'block' : 'none';
                if (analyticsTabEnsembleMethods) analyticsTabEnsembleMethods.style.display = (targetTab === 'ensemble_methods') ? 'block' : 'none';
                if (analyticsTabVerification) analyticsTabVerification.style.display = (targetTab === 'verification') ? 'block' : 'none';

                setTimeout(() => {
                    if (lastAnalyticsDataCache) {
                        renderCharts(lastAnalyticsDataCache);
                    }
                }, 30);
            });
        });
    }

    // Per-class Metric Pill Group Event Listener (Base Models)
    const perClassMetricGroup = document.getElementById('perClassMetricGroup');
    const perClassChartTitle = document.getElementById('perClassChartTitle');

    if (perClassMetricGroup) {
        const metricPills = perClassMetricGroup.querySelectorAll('.metric-pill');
        metricPills.forEach(pill => {
            pill.addEventListener('click', () => {
                metricPills.forEach(p => p.classList.remove('active'));
                pill.classList.add('active');
                
                currentPerClassMetric = pill.getAttribute('data-metric');

                const metricDisplayNames = {
                    'f1_score': 'F1-Score',
                    'precision': 'Precision',
                    'recall': 'Recall'
                };
                
                if (perClassChartTitle) {
                    perClassChartTitle.textContent = `Per-Class Disease ${metricDisplayNames[currentPerClassMetric] || 'Metric'} Comparison Across Models (%)`;
                }

                if (lastAnalyticsDataCache) {
                    renderCharts(lastAnalyticsDataCache);
                }
            });
        });
    }

    // Per-class Metric Pill Group Event Listener (Ensemble Methods)
    const ensPerClassMetricGroup = document.getElementById('ensPerClassMetricGroup');
    const ensPerClassChartTitle = document.getElementById('ensPerClassChartTitle');

    if (ensPerClassMetricGroup) {
        const ensMetricPills = ensPerClassMetricGroup.querySelectorAll('.metric-pill');
        ensMetricPills.forEach(pill => {
            pill.addEventListener('click', () => {
                ensMetricPills.forEach(p => p.classList.remove('active'));
                pill.classList.add('active');
                
                currentEnsPerClassMetric = pill.getAttribute('data-metric');

                const metricDisplayNames = {
                    'f1_score': 'F1-Score',
                    'precision': 'Precision',
                    'recall': 'Recall'
                };
                
                if (ensPerClassChartTitle) {
                    ensPerClassChartTitle.textContent = `Per-Class Disease ${metricDisplayNames[currentEnsPerClassMetric] || 'Metric'} Comparison Across Ensemble Methods (%)`;
                }

                if (lastAnalyticsDataCache) {
                    renderCharts(lastAnalyticsDataCache);
                }
            });
        });
    }

    if (analyticsToggleBtn && analyticsSection) {
        analyticsToggleBtn.addEventListener('click', () => {
            const isHidden = window.getComputedStyle(analyticsSection).display === 'none' || analyticsSection.style.display === 'none';
            if (isHidden) {
                analyticsSection.style.display = 'block';
                analyticsToggleBtn.classList.add('active');
                setTimeout(() => {
                    if (!analyticsLoaded) {
                        loadAndRenderAnalytics();
                    } else if (lastAnalyticsDataCache) {
                        renderCharts(lastAnalyticsDataCache);
                    }
                }, 50);
            } else {
                analyticsSection.style.display = 'none';
                analyticsToggleBtn.classList.remove('active');
            }
        });
    }

    if (analyticsCloseBtn && analyticsSection) {
        analyticsCloseBtn.addEventListener('click', () => {
            analyticsSection.style.display = 'none';
            if (analyticsToggleBtn) {
                analyticsToggleBtn.classList.remove('active');
            }
        });
    }

    async function loadAndRenderAnalytics() {
        try {
            const response = await fetch(`/api/v1/analytics?protocol=${currentProtocol}`);
            const data = await response.json();
            if (data.status === 'success') {
                analyticsLoaded = true;
                lastAnalyticsDataCache = data;
                const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
                const isLight = currentTheme === 'light';
                renderTablesAndVerification(data, isLight);
                renderCharts(data);
            }
        } catch (err) {
            console.error('Failed to load analytics:', err);
        }
    }

    function renderCharts(data) {
        if (!data) return;

        const baseModels = data.base_models || {};
        const ensembleModels = data.ensemble_models || {};
        const modelKeys = Object.keys(baseModels);
        const modelNames = modelKeys.map(k => baseModels[k]?.name || k);
        const classNames = data.class_names || [];
        const classDisplayMap = data.class_display_names || {};
        const displayClassLabels = classNames.map(c => classDisplayMap[c] || c.replace(/_/g, ' '));

        const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
        const isLight = currentTheme === 'light';
        const textColor = isLight ? '#0f172a' : '#f8fafc';
        const subTextColor = isLight ? '#475569' : '#94a3b8';
        const gridColor = isLight ? 'rgba(15, 23, 42, 0.08)' : 'rgba(255, 255, 255, 0.08)';

        const colors = [
            { bg: 'rgba(99, 102, 241, 0.85)', border: '#6366f1' },  // Electric Indigo (ResNet-50)
            { bg: 'rgba(6, 182, 212, 0.85)', border: '#06b6d4' },   // Vivid Cyan (DenseNet-121)
            { bg: 'rgba(16, 185, 129, 0.85)', border: '#10b981' },  // Bright Emerald (EfficientNet-B0)
            { bg: 'rgba(244, 63, 94, 0.85)', border: '#f43f5e' }    // Neon Rose (Swin Tiny)
        ];

        // Always render tables and text widgets
        renderTablesAndVerification(data, isLight);

        if (typeof Chart === 'undefined') return;

        // 1. Chart Benchmark Bar
        try {
            const ctxBenchmark = document.getElementById('chartBenchmark')?.getContext('2d');
            if (ctxBenchmark && modelKeys.length > 0) {
                if (chartInstanceBenchmark) chartInstanceBenchmark.destroy();

                const accuracies = modelKeys.map(k => baseModels[k]?.accuracy ?? null);
                const precisions = modelKeys.map(k => baseModels[k]?.precision ?? null);
                const recalls = modelKeys.map(k => baseModels[k]?.recall ?? null);
                const f1s = modelKeys.map(k => baseModels[k]?.f1_score ?? null);

                chartInstanceBenchmark = new Chart(ctxBenchmark, {
                    type: 'bar',
                    data: {
                        labels: modelNames,
                        datasets: [
                            { label: 'Accuracy (%)', data: accuracies, backgroundColor: 'rgba(99, 102, 241, 0.85)' },
                            { label: 'Precision (%)', data: precisions, backgroundColor: 'rgba(6, 182, 212, 0.85)' },
                            { label: 'Recall (%)', data: recalls, backgroundColor: 'rgba(16, 185, 129, 0.85)' },
                            { label: 'F1-Score (%)', data: f1s, backgroundColor: 'rgba(244, 63, 94, 0.85)' }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: { min: 80, max: 100, ticks: { color: subTextColor }, grid: { color: gridColor } },
                            x: { ticks: { color: textColor, font: { family: 'Outfit', size: 11, weight: '600' } }, grid: { display: false } }
                        },
                        plugins: {
                            legend: {
                                position: 'bottom',
                                align: 'center',
                                labels: {
                                    color: textColor,
                                    padding: 14,
                                    font: { family: 'Outfit', size: 12, weight: '600' },
                                    usePointStyle: true,
                                    pointStyle: 'circle'
                                }
                            }
                        }
                    }
                });
            }
        } catch (e) {
            console.error('Error rendering chartBenchmark:', e);
        }

        // 2. Chart Loss History
        try {
            const ctxLoss = document.getElementById('chartLossHistory')?.getContext('2d');
            if (ctxLoss && modelKeys.length > 0) {
                if (chartInstanceLoss) chartInstanceLoss.destroy();

                const datasetsLoss = [];
                modelKeys.forEach((k, idx) => {
                    const history = baseModels[k]?.history || [];
                    if (history && history.length > 0) {
                        datasetsLoss.push({
                            label: `${baseModels[k].name}`,
                            data: history.map(h => h.val_loss),
                            borderColor: colors[idx % colors.length].border,
                            backgroundColor: colors[idx % colors.length].bg,
                            borderWidth: 2.5,
                            tension: 0.25,
                            pointRadius: 2,
                            fill: false
                        });
                    }
                });

                const maxEpochs = Math.max(0, ...modelKeys.map(k => (baseModels[k]?.history?.length || 0)));
                const epochLabels = maxEpochs > 0 ? Array.from({ length: maxEpochs }, (_, i) => `Epoch ${i + 1}`) : ['Epoch 1'];

                if (datasetsLoss.length > 0) {
                    chartInstanceLoss = new Chart(ctxLoss, {
                        type: 'line',
                        data: { labels: epochLabels, datasets: datasetsLoss },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {
                                y: { ticks: { color: subTextColor }, grid: { color: gridColor } },
                                x: { ticks: { color: subTextColor }, grid: { display: false } }
                            },
                            plugins: {
                                legend: {
                                    position: 'bottom',
                                    align: 'center',
                                    labels: {
                                        color: textColor,
                                        padding: 14,
                                        font: { family: 'Outfit', size: 12, weight: '600' },
                                        usePointStyle: true,
                                        pointStyle: 'circle'
                                    }
                                }
                            }
                        }
                    });
                }
            }
        } catch (e) {
            console.error('Error rendering chartLossHistory:', e);
        }

        // 3. Chart Per-Class Breakdown
        try {
            const ctxPerClass = document.getElementById('chartPerClass')?.getContext('2d');
            if (ctxPerClass && modelKeys.length > 0) {
                if (chartInstancePerClass) chartInstancePerClass.destroy();

                const datasetsPerClass = modelKeys.map((k, idx) => {
                    const perClassObj = baseModels[k]?.per_class || {};
                    const metricValues = classNames.map(c => perClassObj[c] ? perClassObj[c][currentPerClassMetric] : 0);
                    return {
                        label: baseModels[k].name,
                        data: metricValues,
                        backgroundColor: colors[idx % colors.length].bg,
                        borderColor: colors[idx % colors.length].border,
                        borderWidth: 1.5
                    };
                });

                chartInstancePerClass = new Chart(ctxPerClass, {
                    type: 'bar',
                    data: { labels: displayClassLabels, datasets: datasetsPerClass },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: { min: 80, max: 100, ticks: { color: subTextColor }, grid: { color: gridColor } },
                            x: { ticks: { color: textColor, font: { family: 'Outfit', size: 11, weight: '600' } }, grid: { display: false } }
                        },
                        plugins: {
                            legend: {
                                position: 'bottom',
                                align: 'center',
                                labels: {
                                    color: textColor,
                                    padding: 16,
                                    font: { family: 'Outfit', size: 12, weight: '600' },
                                    usePointStyle: true,
                                    pointStyle: 'circle'
                                }
                            }
                        }
                    }
                });
            }
        } catch (e) {
            console.error('Error rendering chartPerClass:', e);
        }

        // --- ENSEMBLE METHODS CHARTS ---
        const ensKeys = Object.keys(ensembleModels);

        // 4. Chart Ensemble Benchmark Bar
        try {
            const ctxEnsembleBenchmark = document.getElementById('chartEnsembleBenchmark')?.getContext('2d');
            if (ctxEnsembleBenchmark && ensKeys.length > 0) {
                if (chartInstanceEnsembleBenchmark) chartInstanceEnsembleBenchmark.destroy();

                const ensNames = ensKeys.map(k => ensembleModels[k].name);
                const ensAcc = ensKeys.map(k => ensembleModels[k].accuracy);
                const ensPrec = ensKeys.map(k => ensembleModels[k].precision);
                const ensRec = ensKeys.map(k => ensembleModels[k].recall);
                const ensF1 = ensKeys.map(k => ensembleModels[k].f1_score);

                chartInstanceEnsembleBenchmark = new Chart(ctxEnsembleBenchmark, {
                    type: 'bar',
                    data: {
                        labels: ensNames,
                        datasets: [
                            { label: 'Accuracy (%)', data: ensAcc, backgroundColor: 'rgba(99, 102, 241, 0.85)' },
                            { label: 'Precision (%)', data: ensPrec, backgroundColor: 'rgba(6, 182, 212, 0.85)' },
                            { label: 'Recall (%)', data: ensRec, backgroundColor: 'rgba(16, 185, 129, 0.85)' },
                            { label: 'F1-Score (%)', data: ensF1, backgroundColor: 'rgba(244, 63, 94, 0.85)' }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: { min: 94, max: 100, ticks: { color: subTextColor }, grid: { color: gridColor } },
                            x: { ticks: { color: textColor, font: { family: 'Outfit', size: 10, weight: '600' } }, grid: { display: false } }
                        },
                        plugins: {
                            legend: {
                                position: 'bottom',
                                align: 'center',
                                labels: {
                                    color: textColor,
                                    padding: 14,
                                    font: { family: 'Outfit', size: 12, weight: '600' },
                                    usePointStyle: true,
                                    pointStyle: 'circle'
                                }
                            }
                        }
                    }
                });
            }
        } catch (e) {
            console.error('Error rendering chartEnsembleBenchmark:', e);
        }

        // 5. Chart Ensemble Gain (+%)
        try {
            const ctxEnsembleGain = document.getElementById('chartEnsembleGain')?.getContext('2d');
            if (ctxEnsembleGain && ensKeys.length > 0) {
                if (chartInstanceEnsembleGain) chartInstanceEnsembleGain.destroy();

                const ensNames = ensKeys.map(k => ensembleModels[k].name);
                const gains = ensKeys.map(k => ensembleModels[k].improvement);

                chartInstanceEnsembleGain = new Chart(ctxEnsembleGain, {
                    type: 'bar',
                    data: {
                        labels: ensNames,
                        datasets: [{
                            label: 'Accuracy Improvement over Best Single Model (+%)',
                            data: gains,
                            backgroundColor: 'rgba(16, 185, 129, 0.85)',
                            borderColor: '#10b981',
                            borderWidth: 1.5
                        }]
                    },
                    options: {
                        indexAxis: 'y',
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            x: { min: 0, max: 1.0, ticks: { color: subTextColor, callback: v => `+${v}%` }, grid: { color: gridColor } },
                            y: { ticks: { color: textColor, font: { family: 'Outfit', size: 10, weight: '600' } }, grid: { display: false } }
                        },
                        plugins: {
                            legend: {
                                position: 'bottom',
                                labels: { color: textColor, font: { family: 'Outfit', size: 12, weight: '600' } }
                            }
                        }
                    }
                });
            }
        } catch (e) {
            console.error('Error rendering chartEnsembleGain:', e);
        }

        // 6. Chart Ensemble Per-Class Breakdown
        try {
            const ctxEnsemblePerClass = document.getElementById('chartEnsemblePerClass')?.getContext('2d');
            if (ctxEnsemblePerClass && ensKeys.length > 0) {
                if (chartInstanceEnsemblePerClass) chartInstanceEnsemblePerClass.destroy();

                const ensColors = [
                    { bg: 'rgba(99, 102, 241, 0.85)', border: '#6366f1' },  // Indigo (Hard Voting)
                    { bg: 'rgba(6, 182, 212, 0.85)', border: '#06b6d4' },   // Cyan (Soft Voting)
                    { bg: 'rgba(16, 185, 129, 0.85)', border: '#10b981' },  // Emerald (Weighted Voting)
                    { bg: 'rgba(245, 158, 11, 0.85)', border: '#f59e0b' },  // Amber (Stacking LR)
                    { bg: 'rgba(168, 85, 247, 0.85)', border: '#a855f7' },  // Purple (Stacking RF)
                    { bg: 'rgba(244, 63, 94, 0.85)', border: '#f43f5e' }    // Rose (Stacking XGB)
                ];

                const datasetsEnsPerClass = ensKeys.map((k, idx) => {
                    const perClassObj = ensembleModels[k].per_class || {};
                    const metricValues = classNames.map(c => perClassObj[c] ? perClassObj[c][currentEnsPerClassMetric] : 0);
                    return {
                        label: ensembleModels[k].name,
                        data: metricValues,
                        backgroundColor: ensColors[idx % ensColors.length].bg,
                        borderColor: ensColors[idx % ensColors.length].border,
                        borderWidth: 1.5
                    };
                });

                chartInstanceEnsemblePerClass = new Chart(ctxEnsemblePerClass, {
                    type: 'bar',
                    data: { labels: displayClassLabels, datasets: datasetsEnsPerClass },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: { min: 80, max: 100, ticks: { color: subTextColor }, grid: { color: gridColor } },
                            x: { ticks: { color: textColor, font: { family: 'Outfit', size: 11, weight: '600' } }, grid: { display: false } }
                        },
                        plugins: {
                            legend: {
                                position: 'bottom',
                                align: 'center',
                                labels: {
                                    color: textColor,
                                    padding: 14,
                                    font: { family: 'Outfit', size: 11, weight: '600' },
                                    usePointStyle: true,
                                    pointStyle: 'circle'
                                }
                            }
                        }
                    }
                });
            }
        } catch (e) {
            console.error('Error rendering chartEnsemblePerClass:', e);
        }
    }

    function renderTablesAndVerification(data, isLight) {
        if (!data) return;
        const formatPercent = value => Number.isFinite(Number(value)) ? `${Number(value).toFixed(2)}%` : '-';
        const formatMetric = (value, digits = 4) => Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : '-';
        const textColor = isLight ? '#0f172a' : '#f8fafc';
        const subTextColor = isLight ? '#475569' : '#94a3b8';
        const gridColor = isLight ? 'rgba(15, 23, 42, 0.08)' : 'rgba(255, 255, 255, 0.08)';

        const baseModels = data.base_models || {};
        const ensembleModels = data.ensemble_models || {};
        const verif = data.verification || {};

        const protoText = (data.protocol === 'oof') ? '5-Fold OOF Protocol' : 'Single-Split Protocol';
        const baseCardTitle = document.querySelector('#analyticsTabBaseModels .chart-card-full .chart-card-title');
        if (baseCardTitle) {
            baseCardTitle.innerHTML = `<i class="fa-solid fa-table-list" style="color: var(--cyan-color);"></i> Base Models Benchmark Table <span class="table-pill ${data.protocol === 'oof' ? 'green' : 'blue'}" style="margin-left: 8px;">${protoText}</span>`;
        }
        const ensCardTitle = document.querySelector('#analyticsTabEnsembleMethods .chart-card-full .chart-card-title');
        if (ensCardTitle) {
            ensCardTitle.innerHTML = `<i class="fa-solid fa-table-list" style="color: var(--accent-color);"></i> Ensemble Methods Comprehensive Benchmark Table <span class="table-pill ${data.protocol === 'oof' ? 'green' : 'blue'}" style="margin-left: 8px;">${protoText}</span>`;
        }

        // 1. Render Base Models Table (#baseModelsTable tbody)
        const baseTbody = document.querySelector('#baseModelsTable tbody');
        if (baseTbody) {
            baseTbody.innerHTML = '';
            Object.keys(baseModels).forEach(k => {
                const m = baseModels[k];
                if (!m) return;
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td><strong>${m.name || k}</strong></td>
                    <td><strong>${formatPercent(m.accuracy)}</strong></td>
                    <td>${formatPercent(m.f1_score)}</td>
                    <td>${formatPercent(m.precision)}</td>
                    <td>${formatPercent(m.recall)}</td>
                    <td>${formatPercent(m.weighted_f1)}</td>
                    <td>${formatMetric(m.kappa)}</td>
                    <td>${formatMetric(m.ece)}</td>
                    <td>${formatMetric(m.brier)}</td>
                    <td>${formatMetric(m.nll)}</td>
                `;
                baseTbody.appendChild(tr);
            });
        }

        // 2. Render Ensemble Models Table (#ensembleModelsTable tbody)
        const ensTbody = document.querySelector('#ensembleModelsTable tbody');
        if (ensTbody) {
            ensTbody.innerHTML = '';
            Object.keys(ensembleModels).forEach(k => {
                const m = ensembleModels[k];
                if (!m) return;
                const imp = Number.isFinite(Number(m.improvement)) ? Number(m.improvement) : null;
                const gainStr = imp == null ? '-' : (imp > 0 ? `+${imp.toFixed(2)}%` : `${imp.toFixed(2)}%`);
                const gainStyle = imp != null && imp > 0 ? 'color: var(--success-color); font-weight: 700;' : 'color: var(--subtext-color);';
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td><strong>${m.name || k}</strong></td>
                    <td><span class="badge" style="background: rgba(16, 185, 129, 0.15); color: #34d399;">${m.type || 'Ensemble'}</span></td>
                    <td><strong style="color: var(--accent-color);">${formatPercent(m.accuracy)}</strong></td>
                    <td>${formatPercent(m.f1_score)}</td>
                    <td>${formatPercent(m.precision)}</td>
                    <td>${formatPercent(m.recall)}</td>
                    <td style="${gainStyle}">${gainStr}</td>
                    <td>${formatMetric(m.ece)}</td>
                    <td>${formatMetric(m.brier)}</td>
                    <td>${formatMetric(m.nll)}</td>
                `;
                ensTbody.appendChild(tr);
            });
        }

        // 3. Verification Highlight Cards
        const verifBestEceEl = document.getElementById('verifBestEce');
        const verifBestBrierEl = document.getElementById('verifBestBrier');
        const verifBestNllEl = document.getElementById('verifBestNll');
        const verifAmbiguityRatioEl = document.getElementById('verifAmbiguityRatio');
        const calibrationRows = (verif.calibration_table || []).filter(
            row => String(row.Protocol || '').toLowerCase() === String(data.protocol || '').toLowerCase()
        );
        const finiteMinimum = key => {
            const values = calibrationRows.map(row => Number(row[key])).filter(Number.isFinite);
            return values.length ? Math.min(...values).toFixed(4) : 'No provenance-linked result available';
        };
        if (verifBestEceEl) verifBestEceEl.textContent = finiteMinimum('ECE_15_bins');
        if (verifBestBrierEl) verifBestBrierEl.textContent = finiteMinimum('Brier_score');
        if (verifBestNllEl) verifBestNllEl.textContent = finiteMinimum('NLL');
        if (verifAmbiguityRatioEl) verifAmbiguityRatioEl.textContent = verif.ambiguity_ratio || 'No provenance-linked result available';

        // 4. Verification Diversity Summary Bar
        const globDisSingleEl = document.getElementById('globDisSingle');
        const globDisOofEl = document.getElementById('globDisOof');
        if (globDisSingleEl) globDisSingleEl.textContent = verif.global_disagreement_single || 'No provenance-linked result available';
        if (globDisOofEl) globDisOofEl.textContent = verif.global_disagreement_oof || 'No provenance-linked result available';

        // 5. Verification Diversity Table (#diversityTable tbody)
        const divTbody = document.querySelector('#diversityTable tbody');
        if (divTbody) {
            divTbody.innerHTML = '';
            const dRows = verif.diversity_table || [];
            dRows.forEach(r => {
                const tr = document.createElement('tr');
                const pair = r['Model Pair'] || r['model_pair'] || '';
                const disSingle = r['Disagreement Single'] || r['Disagreement Single (%)'] || r['disagreement_single'] || '';
                const disOof = r['Disagreement OOF'] || r['Disagreement OOF (%)'] || r['disagreement_oof'] || '';
                const yuleSingle = r["Yule's Q Single"] || r['yule_q_single'] || '';
                const yuleOof = r["Yule's Q OOF"] || r['yule_q_oof'] || '';
                const kapSingle = r['Kappa Single'] || r['kappa_single'] || '';
                const kapOof = r['Kappa OOF'] || r['kappa_oof'] || '';
                tr.innerHTML = `
                    <td><strong>${pair}</strong></td>
                    <td style="color: #f59e0b; font-weight: 600;">${disSingle}</td>
                    <td style="color: #10b981; font-weight: 600;">${disOof}</td>
                    <td>${yuleSingle}</td>
                    <td>${yuleOof}</td>
                    <td>${kapSingle}</td>
                    <td>${kapOof}</td>
                `;
                divTbody.appendChild(tr);
            });
        }

        // 6. Verification McNemar 2x2 Contingency Matrix & Test Stats
        const mcn = verif.mcnemar || {};
        const cm = mcn.contingency_matrix || {};
        const cntN11 = document.getElementById('cntN11');
        const cntN10 = document.getElementById('cntN10');
        const cntN01 = document.getElementById('cntN01');
        const cntN00 = document.getElementById('cntN00');
        const mcnChi2 = document.getElementById('mcnChi2');
        const mcnPval = document.getElementById('mcnPval');
        const mcnConclusion = document.getElementById('mcnConclusion');

        if (cntN11) cntN11.textContent = cm.n11_both_correct != null ? cm.n11_both_correct.toLocaleString() : '-';
        if (cntN10) cntN10.textContent = cm.n10_single_correct_oof_wrong != null ? cm.n10_single_correct_oof_wrong : '-';
        if (cntN01) cntN01.textContent = cm.n01_oof_correct_single_wrong != null ? cm.n01_oof_correct_single_wrong : '-';
        if (cntN00) cntN00.textContent = cm.n00_both_wrong != null ? cm.n00_both_wrong : '-';
        if (mcnChi2) mcnChi2.textContent = Number.isFinite(Number(mcn.mcnemar_chi2)) ? `chi-square = ${Number(mcn.mcnemar_chi2).toFixed(4)}` : 'No provenance-linked result available';
        if (mcnPval) mcnPval.textContent = Number.isFinite(Number(mcn.exact_binomial_p_value)) ? `p = ${Number(mcn.exact_binomial_p_value).toFixed(4)}` : 'No provenance-linked result available';
        if (mcnConclusion && mcn.conclusion) {
            mcnConclusion.innerHTML = `<i class="fa-solid fa-circle-check"></i> ${mcn.conclusion}`;
        }

        // 7. Render Cross-Protocol & Pairwise Hypothesis Testing Matrix (#mcnemarMatrixTable tbody)
        const mcnTbody = document.querySelector('#mcnemarMatrixTable tbody');
        if (mcnTbody) {
            mcnTbody.innerHTML = '';
            const matrixRows = verif.cross_protocol_matrix || mcn.cross_protocol_matrix || [];
            if (matrixRows.length > 0) {
                matrixRows.forEach(r => {
                    const ensName = r.method || '-';
                    const sAcc = Number.isFinite(Number(r.accuracy_a)) ? `${(Number(r.accuracy_a) * 100).toFixed(3)}%` : '-';
                    const oAcc = Number.isFinite(Number(r.accuracy_b)) ? `${(Number(r.accuracy_b) * 100).toFixed(3)}%` : '-';
                    const rd = Number.isFinite(Number(r.paired_risk_difference))
                        ? `${Number(r.paired_risk_difference).toFixed(6)} [${Number(r.paired_risk_difference_ci_low).toFixed(6)}, ${Number(r.paired_risk_difference_ci_high).toFixed(6)}]`
                        : '-';
                    const hVal = Number.isFinite(Number(r.marginal_accuracy_cohens_h)) ? Number(r.marginal_accuracy_cohens_h).toFixed(3) : '-';
                    const disc = r.discordant != null ? `${r.n10_a_correct_b_wrong} / ${r.n01_a_wrong_b_correct}` : '-';
                    const chi2 = Number.isFinite(Number(r.edwards_corrected_chi_square)) ? Number(r.edwards_corrected_chi_square).toFixed(4) : '-';
                    const pVal = Number.isFinite(Number(r.exact_binomial_p_value)) ? Number(r.exact_binomial_p_value).toFixed(6) : '-';
                    const bonf = r.decision || '-';
                    const power = Number.isFinite(Number(r.supplementary_posthoc_power)) ? Number(r.supplementary_posthoc_power).toFixed(4) : '-';
                    const dec = r.decision || '-';

                    const pBadge = `<span class="badge" style="font-weight: 600;">p = ${pVal}</span>`;

                    const bonfBadge = `<span class="badge" style="font-size: 0.75rem;">${bonf}</span>`;

                    const tr = document.createElement('tr');
                    tr.innerHTML = `
                        <td><strong>${ensName}</strong></td>
                        <td><span style="font-weight: 600; color: var(--primary-color);">${sAcc}</span></td>
                        <td><span style="font-weight: 600; color: var(--secondary-color);">${oAcc}</span></td>
                        <td style="font-family: monospace; font-size: 0.85rem;">${rd}</td>
                        <td style="font-family: monospace; font-size: 0.85rem;">${hVal}</td>
                        <td style="font-family: monospace; font-size: 0.85rem;">${disc}</td>
                        <td style="font-family: monospace; font-size: 0.85rem;">&chi;² = ${chi2}</td>
                        <td>${pBadge}</td>
                        <td>${bonfBadge}</td>
                        <td><span style="font-weight: 600; color: var(--text-muted);">${power}</span></td>
                        <td><span style="font-size: 0.8rem; color: #10b981;"><i class="fa-solid fa-check"></i> ${dec}</span></td>
                    `;
                    mcnTbody.appendChild(tr);
                });
            }
        }

        if (typeof Chart === 'undefined') return;

        // 8. Chart ECE Benchmark (Canvas #chartEceBenchmark)
        try {
            const ctxEce = document.getElementById('chartEceBenchmark')?.getContext('2d');
            if (ctxEce && verif.calibration_table && verif.calibration_table.length > 0) {
                if (chartInstanceEce) chartInstanceEce.destroy();

                const calibRows = verif.calibration_table;
                const modelNamesOrder = [
                    "resnet50", "densenet121", "efficientnet_b0", "swin_tiny",
                    "hard_voting", "soft_voting", "weighted_voting",
                    "stacking_logistic_regression", "stacking_random_forest", "stacking_xgboost"
                ];

                const displayNames = [
                    "ResNet-50",
                    "DenseNet-121",
                    "EfficientNet-B0",
                    "Swin-Tiny",
                    "Hard Voting",
                    "Soft Voting",
                    "Weighted Voting",
                    "Stacking (LR)",
                    "Stacking (RF)",
                    "Stacking (XGB)"
                ];

                const singleEce = [];
                const oofEce = [];

                modelNamesOrder.forEach(m => {
                    const sRow = calibRows.find(r => r.Protocol === "single_split" && r.Method === m);
                    const oRow = calibRows.find(r => r.Protocol === "oof" && r.Method === m);
                    singleEce.push(sRow && Number.isFinite(Number(sRow.ECE_15_bins)) ? Number(sRow.ECE_15_bins) : null);
                    oofEce.push(oRow && Number.isFinite(Number(oRow.ECE_15_bins)) ? Number(oRow.ECE_15_bins) : null);
                });

                chartInstanceEce = new Chart(ctxEce, {
                    type: 'bar',
                    data: {
                        labels: displayNames,
                        datasets: [
                            {
                                label: 'Single-Split Holdout (ECE)',
                                data: singleEce,
                                backgroundColor: 'rgba(245, 158, 11, 0.85)',
                                borderColor: '#f59e0b',
                                borderWidth: 1.5
                            },
                            {
                                label: '5-Fold OOF (ECE)',
                                data: oofEce,
                                backgroundColor: 'rgba(16, 185, 129, 0.85)',
                                borderColor: '#10b981',
                                borderWidth: 1.5
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: { 
                                ticks: { color: subTextColor, callback: v => typeof v === 'number' ? v.toFixed(3) : v }, 
                                grid: { color: gridColor },
                                title: { display: true, text: 'Expected Calibration Error (Lower is Better)', color: subTextColor }
                            },
                            x: { 
                                ticks: { color: textColor, font: { family: 'Outfit', size: 10, weight: '600' } }, 
                                grid: { display: false } 
                            }
                        },
                        plugins: {
                            legend: {
                                position: 'bottom',
                                labels: { color: textColor, font: { family: 'Outfit', size: 11, weight: '600' } }
                            }
                        }
                    }
                });
            }
        } catch (e) {
            console.error('Error rendering chartEceBenchmark:', e);
        }

        // 9. Chart Brier Score & NLL (Canvas #chartBrierNll)
        try {
            const ctxBrierNll = document.getElementById('chartBrierNll')?.getContext('2d');
            if (ctxBrierNll && verif.calibration_table && verif.calibration_table.length > 0) {
                if (chartInstanceBrierNll) chartInstanceBrierNll.destroy();

                const calibRows = verif.calibration_table;
                const isOof = (data.protocol === 'oof');
                const targetProto = isOof ? "oof" : "single_split";

                const ensModels = [
                    "hard_voting", "soft_voting", "weighted_voting",
                    "stacking_logistic_regression", "stacking_random_forest", "stacking_xgboost"
                ];

                const ensLabels = [
                    "Hard Voting",
                    "Soft Voting",
                    "Weighted Voting",
                    "Stacking LR",
                    "Stacking RF",
                    "Stacking XGB"
                ];

                const brierScores = [];
                const nllScores = [];

                ensModels.forEach(m => {
                    const row = calibRows.find(r => r.Protocol === targetProto && r.Method === m);
                    brierScores.push(row && Number.isFinite(Number(row.Brier_score)) ? Number(row.Brier_score) : null);
                    nllScores.push(row && Number.isFinite(Number(row.NLL)) ? Number(row.NLL) : null);
                });

                chartInstanceBrierNll = new Chart(ctxBrierNll, {
                    type: 'bar',
                    data: {
                        labels: ensLabels,
                        datasets: [
                            {
                                label: `Brier Score (${targetProto})`,
                                data: brierScores,
                                backgroundColor: 'rgba(6, 182, 212, 0.85)',
                                borderColor: '#06b6d4',
                                borderWidth: 1.5,
                                yAxisID: 'y'
                            },
                            {
                                label: `Negative Log-Likelihood NLL (${targetProto})`,
                                data: nllScores,
                                backgroundColor: 'rgba(168, 85, 247, 0.85)',
                                borderColor: '#a855f7',
                                borderWidth: 1.5,
                                yAxisID: 'y1'
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                type: 'linear',
                                display: true,
                                position: 'left',
                                ticks: { color: '#06b6d4' },
                                grid: { color: gridColor },
                                title: { display: true, text: 'Brier Score (Lower is Better)', color: '#06b6d4' }
                            },
                            y1: {
                                type: 'linear',
                                display: true,
                                position: 'right',
                                ticks: { color: '#a855f7' },
                                grid: { drawOnChartArea: false },
                                title: { display: true, text: 'NLL (Log-Loss)', color: '#a855f7' }
                            },
                            x: {
                                ticks: { color: textColor, font: { family: 'Outfit', size: 10, weight: '600' } },
                                grid: { display: false }
                            }
                        },
                        plugins: {
                            legend: {
                                position: 'bottom',
                                labels: { color: textColor, font: { family: 'Outfit', size: 11, weight: '600' } }
                            }
                        }
                    }
                });
            }
        } catch (e) {
            console.error('Error rendering chartBrierNll:', e);
        }
    }

    // Custom Dropdown Component Logic
    const customDropdowns = document.querySelectorAll('.custom-dropdown');
    
    customDropdowns.forEach(dropdown => {
        const trigger = dropdown.querySelector('.custom-dropdown-trigger');
        const textSpan = dropdown.querySelector('.custom-dropdown-text');
        const items = dropdown.querySelectorAll('.custom-dropdown-item');
        const hiddenInput = dropdown.querySelector('input[type="hidden"]');

        trigger.addEventListener('click', (e) => {
            e.stopPropagation();
            // Close other open dropdowns
            customDropdowns.forEach(other => {
                if (other !== dropdown) other.classList.remove('open');
            });
            dropdown.classList.toggle('open');
        });

        items.forEach(item => {
            item.addEventListener('click', (e) => {
                e.stopPropagation();
                items.forEach(i => i.classList.remove('selected'));
                item.classList.add('selected');
                
                textSpan.textContent = item.textContent;
                hiddenInput.value = item.getAttribute('data-value');
                dropdown.classList.remove('open');
            });
        });
    });

    // Close dropdowns on outside click
    document.addEventListener('click', () => {
        customDropdowns.forEach(dropdown => dropdown.classList.remove('open'));
    });

    // Mobile & Tablet Collapsible Panel Logic
    const controlPanel = document.getElementById('controlPanel');
    const panelHeaderToggle = document.getElementById('panelHeaderToggle');

    if (panelHeaderToggle && controlPanel) {
        panelHeaderToggle.addEventListener('click', () => {
            if (window.innerWidth <= 1024) {
                controlPanel.classList.toggle('collapsed');
            }
        });

        window.addEventListener('resize', () => {
            if (window.innerWidth > 1024) {
                controlPanel.classList.remove('collapsed');
            }
        });
    }

    // Radio Tab Mode Selection
    radioButtons.forEach(radio => {
        radio.addEventListener('change', (e) => {
            modeOptions.forEach(opt => opt.classList.remove('active'));
            e.target.closest('.mode-option').classList.add('active');

            const val = e.target.value;
            subBaseBox.style.display = val === 'single_base' ? 'block' : 'none';
            subEnsembleBox.style.display = val === 'single_ensemble' ? 'block' : 'none';
        });
    });

    // File Input & Drag and Drop Handling
    dropzone.addEventListener('click', () => fileInput.click());
    
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFiles(e.dataTransfer.files);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFiles(e.target.files);
        }
    });

    // Global Clipboard Paste Event (Ctrl + V)
    window.addEventListener('paste', (e) => {
        const items = (e.clipboardData || (e.originalEvent && e.originalEvent.clipboardData)).items;
        if (!items) return;

        let pastedFiles = [];
        for (let item of items) {
            if (item.kind === 'file' && item.type.startsWith('image/')) {
                const blob = item.getAsFile();
                if (blob) {
                    const timeStr = new Date().getTime();
                    const pastedFile = new File([blob], `pasted_image_${timeStr}.png`, { type: blob.type });
                    pastedFiles.push(pastedFile);
                }
            }
        }

        if (pastedFiles.length > 0) {
            selectedFiles = selectedFiles.concat(pastedFiles);
            fileCount.textContent = `Selected ${selectedFiles.length} file(s) (includes pasted image)`;
            fileCount.style.display = 'block';
            btnRun.disabled = false;
            btnClear.style.display = 'block';

            dropzone.style.borderColor = '#10b981';
            setTimeout(() => { dropzone.style.borderColor = ''; }, 1200);
        }
    });

    function handleFiles(files) {
        selectedFiles = selectedFiles.concat(Array.from(files));
        fileCount.textContent = `Selected ${selectedFiles.length} file(s)`;
        fileCount.style.display = 'block';
        btnRun.disabled = false;
        btnClear.style.display = 'block';
    }

    // Clear Button Handling
    btnClear.addEventListener('click', () => {
        selectedFiles = [];
        fileInput.value = '';
        fileCount.textContent = '';
        fileCount.style.display = 'none';
        btnRun.disabled = true;
        btnClear.style.display = 'none';
        badgeStats.textContent = '0 Images';
        resultsContainer.innerHTML = `
            <p style="color: var(--text-secondary); text-align: center; padding: 4rem 0;">
                Select a mode, upload images, and click "Run Classification Pipeline" to view inference results.
            </p>
        `;
    });

    // Inference Protocol Switcher Logic (5-Fold OOF vs Single-Split vs Dual Mode)
    const inferProtocolSwitcher = document.getElementById('inferProtocolSwitcher');
    const inferProtocolInput = document.getElementById('inferProtocolInput');
    const inferProtocolBadge = document.getElementById('inferProtocolBadge');

    if (inferProtocolSwitcher && inferProtocolInput) {
        const protoBtns = inferProtocolSwitcher.querySelectorAll('.infer-protocol-btn');
        protoBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                protoBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                const selectedProto = btn.getAttribute('data-protocol');
                inferProtocolInput.value = selectedProto;

                if (inferProtocolBadge) {
                    if (selectedProto === 'oof') {
                        inferProtocolBadge.textContent = '5-Fold OOF (20 Models)';
                    } else if (selectedProto === 'single') {
                        inferProtocolBadge.textContent = 'Single-Split (4 Models)';
                    } else if (selectedProto === 'dual') {
                        inferProtocolBadge.textContent = 'Dual Mode (Comparative)';
                    }
                }
            });
        });
    }

    // Pipeline Form Submit API Request
    pipelineForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (selectedFiles.length === 0) return;

        btnRun.disabled = true;
        btnRun.innerHTML = '<span class="spinner"></span> Running Classification Pipeline...';

        const formData = new FormData();
        const modeType = document.querySelector('input[name="mode_type"]:checked').value;
        formData.append('mode_type', modeType);

        let selectedModel = '';
        if (modeType === 'single_base') {
            selectedModel = document.getElementById('singleBaseSelect').value;
        } else if (modeType === 'single_ensemble') {
            selectedModel = document.getElementById('singleEnsembleSelect').value;
        }
        formData.append('selected_model', selectedModel);

        const protocolVal = inferProtocolInput ? inferProtocolInput.value : 'oof';
        formData.append('protocol', protocolVal);

        selectedFiles.forEach(file => formData.append('files', file));

        try {
            const response = await fetch('/api/v1/predict', { method: 'POST', body: formData });
            const data = await response.json();

            if (data.status === 'success') {
                renderResults(data);
            } else {
                alert('Error: ' + (data.detail || data.message));
            }
        } catch (err) {
            alert('Pipeline Execution Error: ' + err.message);
        } finally {
            btnRun.disabled = false;
            btnRun.textContent = 'Run Classification Pipeline';
        }
    });

    // Dynamic Render Function Supporting Single, OOF, and Dual Protocols
    function renderResults(data) {
        const protoLabel = data.protocol === 'dual' ? 'Dual Mode (Single vs. OOF)' : (data.protocol === 'single' ? 'Single-Split' : '5-Fold OOF');
        badgeStats.textContent = `${data.total_images} Image(s) | ${data.execution_time_ms} ms [${protoLabel}]`;
        resultsContainer.innerHTML = '';

        data.results.forEach((item, imgIndex) => {
            const card = document.createElement('div');
            card.className = 'image-card';

            const fileObj = selectedFiles[imgIndex];
            const imgUrl = fileObj ? URL.createObjectURL(fileObj) : '';

            if (data.protocol === 'dual' && item.dual_comparison && item.dual_comparison.length > 0) {
                // Dual Mode Render: Side-by-Side Consensus & Comparative Table
                const matchIcon = item.consensus_match 
                    ? '<span class="dual-badge match"><i class="fa-solid fa-circle-check"></i> Consensus Consensus Matched</span>' 
                    : '<span class="dual-badge diff"><i class="fa-solid fa-triangle-exclamation"></i> Consensus Divergence</span>';

                let tableRows = '';
                item.dual_comparison.forEach(row => {
                    const rowMatch = row.match 
                        ? '<span class="dual-badge match"><i class="fa-solid fa-check"></i> Agree</span>'
                        : '<span class="dual-badge diff"><i class="fa-solid fa-xmark"></i> Differ</span>';
                    const isEnsemble = row.category === 'Ensemble Method';
                    const catBadge = isEnsemble ? '<span class="proto-tag oof">Ensemble</span>' : '<span class="proto-tag single">Base</span>';

                    tableRows += `
                        <tr>
                            <td><strong>${row.name}</strong> ${catBadge}</td>
                            <td><span style="color: var(--cyan-color); font-weight: 600;">${row.single_pred}</span> (${row.single_conf}%)</td>
                            <td><span style="color: var(--accent-color); font-weight: 600;">${row.oof_pred}</span> (${row.oof_conf}%)</td>
                            <td>${rowMatch}</td>
                        </tr>
                    `;
                });

                card.innerHTML = `
                    <div class="image-card-header">
                        <span style="font-weight: 600;"><i class="fa-regular fa-image" style="color: var(--accent-color); margin-right: 0.35rem;"></i> ${item.filename}</span>
                        <div>${matchIcon}</div>
                    </div>
                    <div class="image-card-body" style="display: block;">
                        <div style="display: flex; gap: 1.25rem; margin-bottom: 1rem; align-items: center; flex-wrap: wrap;">
                            <img src="${imgUrl}" class="preview-thumb" style="width: 140px; height: 140px;" alt="${item.filename}">
                            <div style="flex: 1; min-width: 260px;">
                                <div class="dual-consensus-wrapper">
                                    <div class="dual-consensus-card single">
                                        <div class="dual-consensus-card-header">
                                            <span class="dual-consensus-title"><i class="fa-solid fa-bolt"></i> Single-Split Consensus</span>
                                            <span class="proto-tag single">Validation-fitted meta</span>
                                        </div>
                                        <div class="dual-consensus-val" style="color: var(--cyan-color);">${item.single_consensus || 'N/A'}</div>
                                    </div>
                                    <div class="dual-consensus-card oof">
                                        <div class="dual-consensus-card-header">
                                            <span class="dual-consensus-title"><i class="fa-solid fa-arrows-rotate"></i> 5-Fold OOF Consensus</span>
                                            <span class="proto-tag oof">Cross-fitted OOF meta</span>
                                        </div>
                                        <div class="dual-consensus-val" style="color: var(--accent-color);">${item.oof_consensus || 'N/A'}</div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div class="dual-table-wrapper">
                            <table class="dual-table">
                                <thead>
                                    <tr>
                                        <th>Model / Ensemble Method</th>
                                        <th>Single-Split Prediction</th>
                                        <th>5-Fold OOF Prediction (Averaged)</th>
                                        <th>Protocol Agreement</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    ${tableRows}
                                </tbody>
                            </table>
                        </div>
                    </div>
                `;
            } else {
                // Standard Single Protocol Render (OOF or Single)
                let methodsHtml = '';
                for (const [key, pred] of Object.entries(item.predictions)) {
                    const isEnsemble = pred.category === 'Ensemble Method';
                    const itemClass = isEnsemble ? 'method-item ensemble-item' : 'method-item base-item';
                    const confText = typeof pred.confidence_percent === 'number' ? `${pred.confidence_percent}%` : pred.confidence_percent;
                    const barWidth = typeof pred.confidence_percent === 'number' ? pred.confidence_percent : 100;
                    const protoTag = pred.protocol ? `<span class="proto-tag ${pred.protocol.includes('OOF') ? 'oof' : 'single'}" style="float: right; margin-top: -2px;">${pred.protocol}</span>` : '';

                    methodsHtml += `
                        <div class="${itemClass}">
                            <div class="method-name">${pred.name} ${protoTag}</div>
                            <div class="method-pred">${pred.predicted_class}</div>
                            <div style="font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 0.3rem;">Conf: ${confText}</div>
                            <div class="prob-bar-bg">
                                <div class="prob-bar-fill" style="width: ${barWidth}%;"></div>
                            </div>
                        </div>
                    `;
                }

                const consensusText = item.consensus_class ? `Consensus: ${item.consensus_class}` : 'Prediction Complete';

                card.innerHTML = `
                    <div class="image-card-header">
                        <span style="font-weight: 600;"><i class="fa-regular fa-image" style="color: var(--accent-color); margin-right: 0.35rem;"></i> ${item.filename}</span>
                        <span class="consensus-pill">${consensusText}</span>
                    </div>
                    <div class="image-card-body">
                        <img src="${imgUrl}" class="preview-thumb" alt="${item.filename}">
                        <div class="method-grid">
                            ${methodsHtml}
                        </div>
                    </div>
                `;
            }

            resultsContainer.appendChild(card);
        });
    }

    // Pre-fetch analytics in background so charts are instantly available when drawer is toggled
    loadAndRenderAnalytics();
});

