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
    let chartInstanceBenchmark = null;
    let chartInstanceLoss = null;
    let chartInstancePerClass = null;
    let chartInstanceEnsembleBenchmark = null;
    let chartInstanceEnsembleGain = null;
    let chartInstanceEnsemblePerClass = null;
    let chartInstanceBaseVsEnsemble = null;

    // Analytics Tab Switcher (Base Models vs Ensemble Methods)
    const analyticsTabGroup = document.getElementById('analyticsTabGroup');
    const analyticsTabBaseModels = document.getElementById('analyticsTabBaseModels');
    const analyticsTabEnsembleMethods = document.getElementById('analyticsTabEnsembleMethods');

    if (analyticsTabGroup) {
        const tabBtns = analyticsTabGroup.querySelectorAll('.analytics-tab-btn');
        tabBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                tabBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');

                const targetTab = btn.getAttribute('data-tab');
                if (targetTab === 'base_models') {
                    if (analyticsTabBaseModels) analyticsTabBaseModels.style.display = 'block';
                    if (analyticsTabEnsembleMethods) analyticsTabEnsembleMethods.style.display = 'none';
                } else {
                    if (analyticsTabBaseModels) analyticsTabBaseModels.style.display = 'none';
                    if (analyticsTabEnsembleMethods) analyticsTabEnsembleMethods.style.display = 'block';
                }

                if (lastAnalyticsDataCache) {
                    renderCharts(lastAnalyticsDataCache);
                }
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
            const response = await fetch('/api/v1/analytics');
            const data = await response.json();
            if (data.status === 'success') {
                analyticsLoaded = true;
                lastAnalyticsDataCache = data;
                renderCharts(data);
            }
        } catch (err) {
            console.error('Failed to load analytics:', err);
        }
    }

    function renderCharts(data) {
        if (typeof Chart === 'undefined') return;

        const baseModels = data.base_models || {};
        const ensembleModels = data.ensemble_models || {};
        const modelKeys = Object.keys(baseModels);
        const modelNames = modelKeys.map(k => baseModels[k].name);
        const classNames = data.class_names || [];
        const classDisplayMap = data.class_display_names || {};
        const displayClassLabels = classNames.map(c => classDisplayMap[c] || c.replace(/_/g, ' '));

        // Dynamic theme text & grid colors
        const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
        const isLight = currentTheme === 'light';
        const textColor = isLight ? '#0f172a' : '#f8fafc';
        const subTextColor = isLight ? '#475569' : '#94a3b8';
        const gridColor = isLight ? 'rgba(15, 23, 42, 0.08)' : 'rgba(255, 255, 255, 0.08)';

        // Distinct High-Contrast Colors for 4 Backbones
        const colors = [
            { bg: 'rgba(99, 102, 241, 0.85)', border: '#6366f1' },  // Electric Indigo (ResNet-50)
            { bg: 'rgba(6, 182, 212, 0.85)', border: '#06b6d4' },   // Vivid Cyan (DenseNet-121)
            { bg: 'rgba(16, 185, 129, 0.85)', border: '#10b981' },  // Bright Emerald (EfficientNet-B0)
            { bg: 'rgba(244, 63, 94, 0.85)', border: '#f43f5e' }    // Neon Rose (Swin Tiny)
        ];

        // 1. Chart Benchmark Bar
        const ctxBenchmark = document.getElementById('chartBenchmark')?.getContext('2d');
        if (ctxBenchmark && modelKeys.length > 0) {
            if (chartInstanceBenchmark) chartInstanceBenchmark.destroy();

            const accuracies = modelKeys.map(k => baseModels[k].accuracy);
            const precisions = modelKeys.map(k => baseModels[k].precision);
            const recalls = modelKeys.map(k => baseModels[k].recall);
            const f1s = modelKeys.map(k => baseModels[k].f1_score);

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

        // 2. Chart Loss History
        const ctxLoss = document.getElementById('chartLossHistory')?.getContext('2d');
        if (ctxLoss && modelKeys.length > 0) {
            if (chartInstanceLoss) chartInstanceLoss.destroy();

            const datasetsLoss = [];
            modelKeys.forEach((k, idx) => {
                const history = baseModels[k].history;
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

            const maxEpochs = Math.max(...modelKeys.map(k => baseModels[k].history.length));
            const epochLabels = Array.from({ length: maxEpochs }, (_, i) => `Epoch ${i + 1}`);

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

        // 3. Chart Per-Class Breakdown (Dynamic Metric: f1_score, precision, recall)
        const ctxPerClass = document.getElementById('chartPerClass')?.getContext('2d');
        if (ctxPerClass && modelKeys.length > 0) {
            if (chartInstancePerClass) chartInstancePerClass.destroy();

            const datasetsPerClass = modelKeys.map((k, idx) => {
                const perClassObj = baseModels[k].per_class;
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

        // --- ENSEMBLE METHODS CHARTS ---
        const ensKeys = Object.keys(ensembleModels);

        // 4. Chart Ensemble Benchmark Bar
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

        // 5. Chart Ensemble Gain (+%)
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

        // 6. Chart Ensemble Per-Class Breakdown (Dynamic Metric: f1_score, precision, recall)
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

        // 7. Chart Best Base vs Ensemble Comparison
        const ctxBaseVsEnsemble = document.getElementById('chartBaseVsEnsemble')?.getContext('2d');
        if (ctxBaseVsEnsemble && modelKeys.length > 0 && ensKeys.length > 0) {
            if (chartInstanceBaseVsEnsemble) chartInstanceBaseVsEnsemble.destroy();

            const bestBaseKey = Object.keys(baseModels).reduce((a, b) => baseModels[a].accuracy > baseModels[b].accuracy ? a : b, Object.keys(baseModels)[0]);
            const bestEnsKey = ensKeys.reduce((a, b) => ensembleModels[a].accuracy > ensembleModels[b].accuracy ? a : b, ensKeys[0]);

            const bestBase = baseModels[bestBaseKey];
            const bestEns = ensembleModels[bestEnsKey];

            chartInstanceBaseVsEnsemble = new Chart(ctxBaseVsEnsemble, {
                type: 'bar',
                data: {
                    labels: ['Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)'],
                    datasets: [
                        {
                            label: `Best Base Model (${bestBase.name})`,
                            data: [bestBase.accuracy, bestBase.precision, bestBase.recall, bestBase.f1_score],
                            backgroundColor: 'rgba(99, 102, 241, 0.85)',
                            borderColor: '#6366f1',
                            borderWidth: 1.5
                        },
                        {
                            label: `Best Ensemble Method (${bestEns.name})`,
                            data: [bestEns.accuracy, bestEns.precision, bestEns.recall, bestEns.f1_score],
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
                        y: { min: 94, max: 100, ticks: { color: subTextColor }, grid: { color: gridColor } },
                        x: { ticks: { color: textColor, font: { family: 'Outfit', size: 11, weight: '600' } }, grid: { display: false } }
                    },
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: { color: textColor, padding: 16, font: { family: 'Outfit', size: 12, weight: '600' } }
                        }
                    }
                }
            });
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
                Select a mode, upload images, and click "Run Classification Pipeline" to view real-time results.
            </p>
        `;
    });

    // Pipeline Form Submit API Request
    pipelineForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (selectedFiles.length === 0) return;

        btnRun.disabled = true;
        btnRun.innerHTML = '<span class="spinner"></span> Running Pipeline...';

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

    // Dynamic Render Function
    function renderResults(data) {
        badgeStats.textContent = `${data.total_images} Image(s) | ${data.execution_time_ms} ms (${data.mode_type})`;
        resultsContainer.innerHTML = '';

        data.results.forEach((item, imgIndex) => {
            const card = document.createElement('div');
            card.className = 'image-card';

            const fileObj = selectedFiles[imgIndex];
            const imgUrl = fileObj ? URL.createObjectURL(fileObj) : '';

            let methodsHtml = '';
            for (const [key, pred] of Object.entries(item.predictions)) {
                const isEnsemble = pred.category === 'Ensemble Method';
                const itemClass = isEnsemble ? 'method-item ensemble-item' : 'method-item base-item';
                const confText = typeof pred.confidence_percent === 'number' ? `${pred.confidence_percent}%` : pred.confidence_percent;
                const barWidth = typeof pred.confidence_percent === 'number' ? pred.confidence_percent : 100;

                methodsHtml += `
                    <div class="${itemClass}">
                        <div class="method-name">${pred.name} (${pred.category})</div>
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

            resultsContainer.appendChild(card);
        });
    }

    // Pre-fetch analytics in background so charts are instantly available when drawer is toggled
    loadAndRenderAnalytics();
});
