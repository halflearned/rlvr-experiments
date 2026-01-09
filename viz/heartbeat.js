// RLVR Heartbeat Visualization
console.log('[heartbeat.js] Script loaded, version 26');

class HeartbeatViz {
    constructor() {
        this.data = null;
        this.events = [];
        this.minTs = 0;
        this.maxTs = 0;

        // Canvas contexts
        this.timelineCtx = null;
        this.bufferCtx = null;
        this.fatesCtx = null;
        this.metricCharts = {};

        // Zoom state per metric chart (xMin, xMax as fractions 0-1)
        this.metricZoom = {};

        // Shared zoom state for timeline and buffer (they share time axis)
        this.timeZoom = { xMin: 0, xMax: 1 };

        // Scroll offset for fates (how many versions to skip from bottom)
        this.fatesScrollOffset = 0;

        // Metric history
        this.metrics = {
            // Training
            reward: [],
            loss: [],
            grad_norm: [],
            // TorchTitan
            mfu: [],
            train_tps: [],
            memory_pct: [],
            // vLLM
            vllm_tps: [],
            vllm_output_tokens: [],
            vllm_prompt_tokens: [],
            // Debug
            kl_max: [],
            ratio_max: [],
            diff_ref: [],
            diff_rollout: [],
            // Efficiency (padding/truncation)
            seq_padding_pct: [],
            completion_padding_pct: [],
            completion_len_mean: [],
            finish_stop_pct: [],
            finish_length_pct: []
        };

        // Metrics that use log scale
        this.logScaleMetrics = new Set(['grad_norm']);

        // Cumulative histograms for length distributions (binned for efficiency)
        // Bins: 0-49, 50-99, ..., up to 2048+ (42 bins covers 0-2048 in steps of 50)
        this.histograms = {
            seq_lens: new Array(50).fill(0),      // bins of 50 tokens, up to 2500
            completion_lens: new Array(50).fill(0),
            // Padding token count histograms: bins of 10 tokens, up to 1000
            seq_padding_tokens: new Array(100).fill(0),
            completion_padding_tokens: new Array(100).fill(0)
        };
        this.histogramBinSize = 50;
        this.histogramMax = 2500;
        this.paddingBinSize = 10;  // 10-token bins for padding counts

        // Scatter plot data: reward vs completion length
        this.rewardVsLen = [];  // [{reward, len}, ...]

        // Hover state for quantile charts
        this.quantileHover = {
            completion_len_mean: null,  // percentile being hovered (0-1), or null
            seq_padding_tokens: null,
            completion_padding_tokens: null
        };

        // Hover state for tooltips
        this.hoverMetric = null;
        this.hoverX = 0;

        // Timeline hover state
        this.timelineSpans = [];  // [{event, x, y, w, h}, ...] for hit testing
        this.hoveredSpan = null;
        this.tooltipX = 0;
        this.tooltipY = 0;

        this.init();
    }

    init() {
        // Setup file input
        document.getElementById('trace-file').addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) this.loadFile(file);
        });

        // Setup canvases
        this.setupCanvases();

        // Handle resize
        window.addEventListener('resize', () => this.render());

        // Setup verifier toggle (collapsed by default)
        this.verifiersCollapsed = true;
        const verifyToggle = document.getElementById('verify-toggle');
        if (verifyToggle) {
            verifyToggle.classList.add('collapsed');  // Start collapsed
            verifyToggle.addEventListener('click', () => {
                this.verifiersCollapsed = !this.verifiersCollapsed;
                verifyToggle.classList.toggle('collapsed', this.verifiersCollapsed);
                this.updateTimelinePanelHeight();
                this.render();
            });
        }

        // Try to load default trace
        this.loadDefaultTrace();
    }

    setupCanvases() {
        const timelineCanvas = document.getElementById('timeline-canvas');
        const bufferCanvas = document.getElementById('buffer-canvas');
        const fatesCanvas = document.getElementById('fates-canvas');

        this.timelineCtx = timelineCanvas.getContext('2d');
        this.bufferCtx = bufferCanvas.getContext('2d');
        this.fatesCtx = fatesCanvas.getContext('2d');

        // Store fate bar positions for hover detection
        this.fateBars = [];
        this.hoveredFate = null;

        // Fates canvas hover handling
        fatesCanvas.addEventListener('mousemove', (e) => {
            const rect = fatesCanvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            this.tooltipX = e.clientX;
            this.tooltipY = e.clientY;

            // Find fate bar under cursor
            let found = null;
            for (const bar of this.fateBars) {
                if (x >= bar.x && x <= bar.x + bar.w && y >= bar.y && y <= bar.y + bar.h) {
                    found = bar;
                    break;
                }
            }

            if (found !== this.hoveredFate) {
                this.hoveredFate = found;
                this.updateFatesTooltip();
            }
        });

        fatesCanvas.addEventListener('mouseleave', () => {
            this.hoveredFate = null;
            this.updateFatesTooltip();
        });

        // Fates scroll handling
        fatesCanvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const delta = Math.sign(e.deltaY);
            this.fatesScrollOffset = Math.max(0, this.fatesScrollOffset + delta);
            this.renderFates();
        }, { passive: false });

        // Timeline zoom/pan handling
        timelineCanvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            this.handleTimeZoom(e, timelineCanvas);
        }, { passive: false });

        timelineCanvas.addEventListener('dblclick', () => {
            this.timeZoom = { xMin: 0, xMax: 1 };
            this.renderTimeline();
            this.renderBuffer();
        });

        // Buffer zoom/pan handling (synced with timeline)
        bufferCanvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            this.handleTimeZoom(e, bufferCanvas);
        }, { passive: false });

        bufferCanvas.addEventListener('dblclick', () => {
            this.timeZoom = { xMin: 0, xMax: 1 };
            this.renderTimeline();
            this.renderBuffer();
        });

        // Timeline hover handling for span tooltips
        timelineCanvas.addEventListener('mousemove', (e) => {
            const rect = timelineCanvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            this.tooltipX = e.clientX;
            this.tooltipY = e.clientY;

            // Find span under cursor
            let found = null;
            for (const span of this.timelineSpans) {
                if (x >= span.x && x <= span.x + span.w && y >= span.y && y <= span.y + span.h) {
                    found = span;
                    break;
                }
            }

            if (found !== this.hoveredSpan) {
                this.hoveredSpan = found;
                this.updateTimelineTooltip();
            }
        });

        timelineCanvas.addEventListener('mouseleave', () => {
            this.hoveredSpan = null;
            this.updateTimelineTooltip();
        });

        // Timeline click handling for rollout inspection
        timelineCanvas.addEventListener('click', (e) => {
            if (this.hoveredSpan && this.hoveredSpan.event.name === 'verify') {
                const event = this.hoveredSpan.event;
                const promptId = event.prompt_id;
                const version = event.version;
                if (promptId && promptId !== 'unknown') {
                    this.showRolloutModal(promptId, version);
                }
            }
        });

        // Setup metric mini-charts with hover handling
        const metricNames = Object.keys(this.metrics);
        metricNames.forEach(metric => {
            const canvas = document.getElementById(`chart-${metric}`);
            if (canvas) {
                this.metricCharts[metric] = canvas.getContext('2d');

                // Initialize zoom state
                this.metricZoom[metric] = { xMin: 0, xMax: 1 };

                // Add hover listeners
                canvas.addEventListener('mousemove', (e) => {
                    const rect = canvas.getBoundingClientRect();
                    // Quantile charts use their own hover logic
                    if (metric === 'completion_len_mean') {
                        this.quantileHover.completion_len_mean = (e.clientX - rect.left) / rect.width;
                        this.renderQuantileChart('completion_len_mean', this.histograms.completion_lens, this.histogramBinSize, '', 'completion_len_mean');
                    } else if (metric === 'seq_padding_pct') {
                        this.quantileHover.seq_padding_tokens = (e.clientX - rect.left) / rect.width;
                        this.renderQuantileChart('seq_padding_pct', this.histograms.seq_padding_tokens, this.paddingBinSize, ' toks', 'seq_padding_tokens');
                    } else if (metric === 'completion_padding_pct') {
                        this.quantileHover.completion_padding_tokens = (e.clientX - rect.left) / rect.width;
                        this.renderQuantileChart('completion_padding_pct', this.histograms.completion_padding_tokens, this.paddingBinSize, ' toks', 'completion_padding_tokens');
                    } else {
                        this.hoverMetric = metric;
                        this.hoverX = (e.clientX - rect.left) / rect.width;
                        this.renderMetricChart(metric);
                    }
                });
                canvas.addEventListener('mouseleave', () => {
                    if (metric === 'completion_len_mean') {
                        this.quantileHover.completion_len_mean = null;
                        this.renderQuantileChart('completion_len_mean', this.histograms.completion_lens, this.histogramBinSize, '', 'completion_len_mean');
                    } else if (metric === 'seq_padding_pct') {
                        this.quantileHover.seq_padding_tokens = null;
                        this.renderQuantileChart('seq_padding_pct', this.histograms.seq_padding_tokens, this.paddingBinSize, ' toks', 'seq_padding_tokens');
                    } else if (metric === 'completion_padding_pct') {
                        this.quantileHover.completion_padding_tokens = null;
                        this.renderQuantileChart('completion_padding_pct', this.histograms.completion_padding_tokens, this.paddingBinSize, ' toks', 'completion_padding_tokens');
                    } else {
                        this.hoverMetric = null;
                        this.renderMetricChart(metric);
                    }
                });

                // Add zoom/pan with wheel
                canvas.addEventListener('wheel', (e) => {
                    e.preventDefault();
                    const rect = canvas.getBoundingClientRect();
                    const mouseX = (e.clientX - rect.left) / rect.width;  // 0-1 position in canvas

                    const zoom = this.metricZoom[metric];
                    const currentRange = zoom.xMax - zoom.xMin;

                    if (e.ctrlKey || e.metaKey || Math.abs(e.deltaY) > Math.abs(e.deltaX)) {
                        // Zoom: ctrl+wheel or vertical scroll
                        const zoomFactor = e.deltaY > 0 ? 1.15 : 0.87;  // zoom out / zoom in
                        const newRange = Math.min(1, Math.max(0.02, currentRange * zoomFactor));

                        // Zoom centered on mouse position
                        const mouseDataX = zoom.xMin + mouseX * currentRange;
                        const newMin = mouseDataX - mouseX * newRange;
                        const newMax = mouseDataX + (1 - mouseX) * newRange;

                        // Clamp to [0, 1]
                        if (newMin < 0) {
                            zoom.xMin = 0;
                            zoom.xMax = Math.min(1, newRange);
                        } else if (newMax > 1) {
                            zoom.xMax = 1;
                            zoom.xMin = Math.max(0, 1 - newRange);
                        } else {
                            zoom.xMin = newMin;
                            zoom.xMax = newMax;
                        }
                    } else {
                        // Pan: horizontal scroll (two-finger swipe on trackpad)
                        const panAmount = (e.deltaX / rect.width) * currentRange * 0.5;
                        const newMin = zoom.xMin + panAmount;
                        const newMax = zoom.xMax + panAmount;

                        if (newMin >= 0 && newMax <= 1) {
                            zoom.xMin = newMin;
                            zoom.xMax = newMax;
                        } else if (newMin < 0) {
                            zoom.xMin = 0;
                            zoom.xMax = currentRange;
                        } else {
                            zoom.xMax = 1;
                            zoom.xMin = 1 - currentRange;
                        }
                    }

                    this.renderMetricChart(metric);
                }, { passive: false });

                // Double-click to reset zoom
                canvas.addEventListener('dblclick', () => {
                    this.metricZoom[metric] = { xMin: 0, xMax: 1 };
                    this.renderMetricChart(metric);
                });
            }
        });
    }

    async loadDefaultTrace() {
        try {
            // Try /trace first (served by serve.py with configured trace file)
            // Then fall back to common file locations
            const paths = [
                '/trace',  // serve.py endpoint (latest or specified trace)
                '../traces/trace.jsonl',
                'traces/trace.jsonl',
                '../tracers/trace.jsonl',
                'tracers/trace.jsonl',
                '../trace.jsonl',
                '/efs/rlvr-experiments/traces/trace.jsonl'
            ];

            for (const path of paths) {
                try {
                    const response = await fetch(path);
                    if (response.ok) {
                        const text = await response.text();
                        this.processJSONL(text);
                        console.log(`Loaded trace from ${path}`);
                        return;
                    }
                } catch (e) {
                    continue;
                }
            }

            console.log('No default trace found. Please load a trace file.');
        } catch (e) {
            console.log('No default trace found:', e);
        }
    }

    loadFile(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const text = e.target.result;
                // Try JSONL first, fall back to old Chrome format
                if (text.trim().startsWith('{') && !text.trim().startsWith('{"traceEvents"')) {
                    this.processJSONL(text);
                } else {
                    // Legacy Chrome format
                    const data = JSON.parse(text);
                    this.processLegacyData(data);
                }
            } catch (err) {
                console.error('Failed to parse trace file:', err);
            }
        };
        reader.readAsText(file);
    }

    processJSONL(text) {
        console.log('[processJSONL] Parsing', text.length, 'chars');
        // Parse JSONL format
        this.events = [];
        for (const line of text.split('\n')) {
            if (line.trim()) {
                try {
                    this.events.push(JSON.parse(line));
                } catch (e) {
                    console.warn('Skipping invalid JSON line:', line);
                }
            }
        }

        if (this.events.length === 0) return;

        // Find time range (ts is already in seconds)
        const tsEvents = this.events.filter(e => e.ts !== undefined);
        this.minTs = 0; // JSONL timestamps are relative to start
        this.maxTs = Math.max(...tsEvents.map(e => e.ts + (e.dur || 0)));

        // Extract metrics
        this.extractMetrics();
        this.updateHeaderStats();
        this.render();
    }

    processLegacyData(data) {
        // Handle old Chrome tracing format for backwards compatibility
        this.data = data;
        const rawEvents = data.traceEvents || [];

        // Convert to new format
        this.events = rawEvents.map(e => {
            if (e.ph === 'X') {
                return { type: 'span', name: e.name, ts: e.ts / 1e6, dur: e.dur / 1e6, ...e.args };
            } else if (e.ph === 'C') {
                return { type: 'counter', name: e.name, ts: e.ts / 1e6, ...e.args };
            } else if (e.ph === 'i') {
                return { type: 'instant', name: e.name, ts: e.ts / 1e6, ...e.args };
            }
            return { type: 'meta', ...e };
        });

        const tsEvents = this.events.filter(e => e.ts !== undefined);
        if (tsEvents.length === 0) return;

        this.minTs = 0;
        this.maxTs = Math.max(...tsEvents.map(e => e.ts + (e.dur || 0)));

        this.extractMetrics();
        this.updateHeaderStats();
        this.render();
    }

    extractMetrics() {
        this.metrics = {
            reward: [], loss: [], grad_norm: [],
            mfu: [], train_tps: [], memory_pct: [],
            vllm_tps: [], vllm_output_tokens: [], vllm_prompt_tokens: [],
            kl_max: [], ratio_max: [], diff_ref: [], diff_rollout: [],
            // Efficiency (padding/truncation)
            seq_padding_pct: [], completion_padding_pct: [], completion_len_mean: [],
            finish_stop_pct: [], finish_length_pct: []
        };
        this.bufferEvents = [];
        this.syncEvents = [];

        // Reset cumulative histograms
        this.histograms = {
            seq_lens: new Array(50).fill(0),
            completion_lens: new Array(50).fill(0),
            seq_padding_tokens: new Array(100).fill(0),
            completion_padding_tokens: new Array(100).fill(0)
        };

        // Reset scatter data
        this.rewardVsLen = [];

        // For utilization tracking
        this.vllmSpans = [];  // {replica, ts, dur}
        this.trainerSpans = [];  // {ts, dur}
        this.verifierSpans = [];  // {worker, ts, dur, n, passed}

        // Metadata from trace (for dynamic lane configuration)
        this.numVllmReplicas = 0;  // Detected from spans
        this.numVerifierWorkers = 0;  // From meta event or detected from spans
        this.verifierMaxConcurrent = 0;  // From meta event (max_concurrent containers per worker)

        let currentVersion = 0;

        for (const event of this.events) {
            const ts = event.ts;  // Already in seconds in new format

            // Counter events (new format: type='counter')
            if (event.type === 'counter') {
                if (event.name === 'rewards') {
                    this.metrics.reward.push({ ts, value: event.mean || 0 });
                }
                if (event.name === 'metrics') {
                    if (event.loss !== undefined) {
                        this.metrics.loss.push({ ts, value: event.loss });
                    }
                    if (event.grad_norm !== undefined) {
                        this.metrics.grad_norm.push({ ts, value: event.grad_norm });
                    }
                }
                if (event.name === 'titan.metrics') {
                    if (event.mfu !== undefined) {
                        this.metrics.mfu.push({ ts, value: event.mfu });
                    }
                    if (event.tps !== undefined) {
                        this.metrics.train_tps.push({ ts, value: event.tps });
                    }
                    if (event.memory_pct !== undefined) {
                        this.metrics.memory_pct.push({ ts, value: event.memory_pct });
                    }
                }
                if (event.name === 'vllm.metrics') {
                    if (event.gen_tps !== undefined) {
                        this.metrics.vllm_tps.push({ ts, value: event.gen_tps });
                    }
                    if (event.output_tokens !== undefined) {
                        this.metrics.vllm_output_tokens.push({ ts, value: event.output_tokens });
                    }
                    if (event.prompt_tokens !== undefined) {
                        this.metrics.vllm_prompt_tokens.push({ ts, value: event.prompt_tokens });
                    }
                }
                if (event.name === 'grpo.debug') {
                    if (event.kl_max !== undefined) {
                        this.metrics.kl_max.push({ ts, value: event.kl_max });
                    }
                    if (event.ratio_max !== undefined) {
                        this.metrics.ratio_max.push({ ts, value: event.ratio_max });
                    }
                    if (event.diff_trainer_ref_max !== undefined) {
                        this.metrics.diff_ref.push({ ts, value: event.diff_trainer_ref_max });
                    }
                    if (event.diff_trainer_rollout_max !== undefined) {
                        this.metrics.diff_rollout.push({ ts, value: event.diff_trainer_rollout_max });
                    }
                }
                // Efficiency: padding stats
                if (event.name === 'batch.padding') {
                    console.log('[extractMetrics] batch.padding event:', 'seq_lens' in event, 'completion_lens' in event);
                    if (event.seq_padding_pct !== undefined) {
                        this.metrics.seq_padding_pct.push({ ts, value: event.seq_padding_pct });
                    }
                    if (event.completion_padding_pct !== undefined) {
                        this.metrics.completion_padding_pct.push({ ts, value: event.completion_padding_pct });
                    }
                    // Accumulate raw lengths into histograms for cumulative quantiles
                    if (event.seq_lens && Array.isArray(event.seq_lens)) {
                        for (const len of event.seq_lens) {
                            const bin = Math.min(Math.floor(len / this.histogramBinSize), this.histograms.seq_lens.length - 1);
                            this.histograms.seq_lens[bin]++;
                        }
                    }
                    if (event.completion_lens && Array.isArray(event.completion_lens)) {
                        for (const len of event.completion_lens) {
                            const bin = Math.min(Math.floor(len / this.histogramBinSize), this.histograms.completion_lens.length - 1);
                            this.histograms.completion_lens[bin]++;
                        }
                        // Also track mean completion length for the sparkline
                        const mean = event.completion_lens.reduce((a, b) => a + b, 0) / event.completion_lens.length;
                        this.metrics.completion_len_mean.push({ ts, value: mean });
                    }
                    // Accumulate per-sample padding token counts into histograms
                    if (event.seq_padding_tokens && Array.isArray(event.seq_padding_tokens)) {
                        for (const tokens of event.seq_padding_tokens) {
                            const bin = Math.min(Math.floor(tokens / this.paddingBinSize), this.histograms.seq_padding_tokens.length - 1);
                            this.histograms.seq_padding_tokens[bin]++;
                        }
                    }
                    if (event.completion_padding_tokens && Array.isArray(event.completion_padding_tokens)) {
                        for (const tokens of event.completion_padding_tokens) {
                            const bin = Math.min(Math.floor(tokens / this.paddingBinSize), this.histograms.completion_padding_tokens.length - 1);
                            this.histograms.completion_padding_tokens[bin]++;
                        }
                    }
                }
                // Reward vs completion length scatter data
                if (event.name === 'batch.reward_vs_len') {
                    if (event.rewards && event.completion_lens) {
                        const rewards = event.rewards;
                        const lens = event.completion_lens;
                        const n = Math.min(rewards.length, lens.length);
                        for (let i = 0; i < n; i++) {
                            this.rewardVsLen.push({ reward: rewards[i], len: lens[i] });
                        }
                    }
                }
                // Efficiency: finish reason distribution
                if (event.name === 'batch.finish_reasons') {
                    // Calculate percentages from counts
                    const stop = event.stop || 0;
                    const length = event.length || 0;
                    const total = stop + length + (event.abort || 0) + (event.unknown || 0);
                    if (total > 0) {
                        this.metrics.finish_stop_pct.push({ ts, value: 100 * stop / total });
                        this.metrics.finish_length_pct.push({ ts, value: 100 * length / total });
                    }
                }
            }

            // Span events for timeline
            if (event.type === 'span') {
                // Track sync events for version changes
                if (event.name === 'sync.titan_to_vllm') {
                    currentVersion++;
                    this.syncEvents.push({ ts, version: currentVersion });
                }
                // Track vLLM generation spans (prefer vllm.generate_single which has replica info)
                if (event.name === 'vllm.generate_single' || event.name === 'vllm.generate' ||
                    (event.name === 'generate' && event.replica !== undefined)) {
                    this.vllmSpans.push({ replica: event.replica || 0, ts: event.ts, dur: event.dur || 0 });
                }
                // Track trainer spans
                if (event.name === 'forward_backward') {
                    this.trainerSpans.push({ ts: event.ts, dur: event.dur || 0 });
                }
                // Track verifier spans
                if (event.name === 'verify') {
                    this.verifierSpans.push({
                        worker: event.worker,
                        slot: event.slot !== undefined ? event.slot : 0,  // slot within worker
                        ts: event.ts,
                        dur: event.dur || 0,
                        passed: event.passed || 0
                    });
                }
            }

            // Meta events for configuration
            if (event.type === 'meta') {
                if (event.verifier_workers !== undefined) {
                    this.numVerifierWorkers = event.verifier_workers;
                }
                if (event.verifier_max_concurrent !== undefined) {
                    this.verifierMaxConcurrent = event.verifier_max_concurrent;
                }
                // Run configuration metadata
                if (event.config_file !== undefined) {
                    this.runConfig = {
                        configFile: event.config_file,
                        runName: event.run_name || '',
                        modelPath: event.model_path || '',
                        dataset: event.dataset || '',
                        vllmBatchSize: event.vllm_batch_size || 0,
                        promptsPerBatch: event.prompts_per_batch || 0,
                        nCompletions: event.n_completions || 1,
                        titanParallelism: event.titan_parallelism || {},
                        startTime: event.start_time || '',
                    };
                }
            }

            // Buffer events (new specialized type)
            if (event.type === 'buffer') {
                this.bufferEvents.push({
                    ts,
                    size: event.size,
                    byVersion: event.by_version || {},
                    fates: event.fates || { used: {}, wasted: {}, partial: {} },
                    version: currentVersion
                });
            }
        }

        // Compute actual replica/worker counts from span data
        const vllmReplicas = new Set(this.vllmSpans.map(s => s.replica));
        this.numVllmReplicas = Math.max(this.numVllmReplicas, vllmReplicas.size);

        const verifierWorkers = new Set(this.verifierSpans.map(s => s.worker));
        this.numVerifierWorkers = Math.max(this.numVerifierWorkers, verifierWorkers.size);

        // Detect max_concurrent from slot field if not in metadata
        if (this.verifierMaxConcurrent === 0 && this.verifierSpans.length > 0) {
            const maxSlot = Math.max(...this.verifierSpans.map(s => s.slot));
            this.verifierMaxConcurrent = maxSlot + 1;  // slots are 0-indexed
        }

        // Update timeline panel height based on number of lanes
        this.updateTimelinePanelHeight();
    }

    /**
     * Dynamically adjust timeline panel height based on number of lanes.
     */
    updateTimelinePanelHeight() {
        // Calculate number of lanes: vLLM replicas + Reference + Trainer + Sync + Verifier (workers × slots)
        const numVllm = Math.max(1, this.numVllmReplicas);  // At least 1
        const numVerifierSlots = this.verifiersCollapsed ? 0 : this.numVerifierWorkers * Math.max(1, this.verifierMaxConcurrent);
        const fixedLanes = 3;  // Reference, Trainer, Sync
        const totalLanes = numVllm + fixedLanes + numVerifierSlots;

        // Base height per lane (pixels), with min/max bounds
        const heightPerLane = 28;
        const minHeight = 200;
        const maxHeight = 650;
        const targetHeight = Math.min(maxHeight, Math.max(minHeight, totalLanes * heightPerLane + 50));

        // Update the CSS grid row height for the timeline panel (index 1, after config panel)
        const container = document.querySelector('.container');
        if (container) {
            const currentRows = container.style.gridTemplateRows || '100px 325px 200px 180px 180px 180px 180px 180px 200px';
            const rowParts = currentRows.split(' ');
            rowParts[1] = `${targetHeight}px`;  // Timeline panel is now index 1 (after config panel)
            container.style.gridTemplateRows = rowParts.join(' ');
        }
    }

    /**
     * Compute quantiles from a binned histogram.
     * @param {number[]} histogram - Array of bin counts
     * @param {number[]} quantiles - Array of quantiles to compute (e.g., [0.5, 0.9, 0.99])
     * @returns {number[]} - Array of values at each quantile (bin midpoints)
     */
    computeQuantiles(histogram, quantiles) {
        const total = histogram.reduce((a, b) => a + b, 0);
        if (total === 0) return quantiles.map(() => 0);

        const results = [];
        for (const q of quantiles) {
            const target = q * total;
            let cumulative = 0;
            for (let i = 0; i < histogram.length; i++) {
                cumulative += histogram[i];
                if (cumulative >= target) {
                    // Return bin midpoint
                    results.push((i + 0.5) * this.histogramBinSize);
                    break;
                }
            }
            if (results.length < quantiles.indexOf(q) + 1) {
                // Fell through, use last bin
                results.push((histogram.length - 0.5) * this.histogramBinSize);
            }
        }
        return results;
    }

    updateHeaderStats() {
        // Find latest values from new format
        const epochs = this.events.filter(e => e.type === 'counter' && e.name === 'epoch');
        const lastEpoch = epochs[epochs.length - 1];

        const steps = this.events.filter(e => e.type === 'span' && e.name === 'forward_backward');

        document.getElementById('current-step').textContent = steps.length || '-';
        document.getElementById('current-epoch').textContent = lastEpoch?.epoch ?? '-';
        document.getElementById('current-version').textContent = this.syncEvents.length || '0';
        const lastBuffer = this.bufferEvents[this.bufferEvents.length - 1];
        document.getElementById('buffer-size').textContent = lastBuffer?.size ?? '-';
    }

    render() {
        this.renderRunConfig();
        this.renderTimeline();
        this.renderBuffer();
        this.renderFates();
        this.renderMetrics();
        this.renderEfficiencyQuantiles();
        this.renderRecommendations();
        this.renderRewardVsLen();
    }

    renderRunConfig() {
        // Update run configuration panel
        if (!this.runConfig) return;

        const cfg = this.runConfig;

        // Basic config values
        const setEl = (id, val) => {
            const el = document.getElementById(id);
            if (el) el.textContent = val || '-';
        };

        // Format start time nicely (ISO to readable)
        let startTimeStr = cfg.startTime || '-';
        if (cfg.startTime) {
            try {
                const d = new Date(cfg.startTime);
                startTimeStr = d.toLocaleString('en-US', {
                    month: 'short', day: 'numeric',
                    hour: '2-digit', minute: '2-digit', hour12: false
                });
            } catch (e) {
                startTimeStr = cfg.startTime;
            }
        }
        setEl('config-start-time', startTimeStr);

        setEl('config-file', cfg.configFile);
        setEl('config-run-name', cfg.runName);

        // Model: show just the model name from path
        const modelName = cfg.modelPath ? cfg.modelPath.split('/').pop() : '-';
        setEl('config-model', modelName);

        setEl('config-dataset', cfg.dataset);
        setEl('config-vllm-batch', cfg.vllmBatchSize ? String(cfg.vllmBatchSize) : '-');

        // Train batch: "X prompts × Y completions"
        if (cfg.promptsPerBatch && cfg.nCompletions) {
            setEl('config-train-batch', `${cfg.promptsPerBatch} × ${cfg.nCompletions}`);
        }

        // Parallelism info - dynamically add items for each Titan role
        const configGrid = document.getElementById('run-config');
        if (configGrid && cfg.titanParallelism) {
            // Remove any previously added parallelism items
            const oldItems = configGrid.querySelectorAll('.parallelism-item');
            oldItems.forEach(el => el.remove());

            for (const [role, p] of Object.entries(cfg.titanParallelism)) {
                const item = document.createElement('div');
                item.className = 'config-item parallelism-item';

                // Format: "dp=2, tp=4" (omit dp_replicate if 1)
                let parts = [];
                if (p.dp_replicate > 1) parts.push(`dp_rep=${p.dp_replicate}`);
                if (p.dp_shard > 1) parts.push(`dp=${p.dp_shard}`);
                if (p.tp > 1) parts.push(`tp=${p.tp}`);
                const parallelStr = parts.length > 0 ? parts.join(', ') : 'none';

                // Capitalize role name
                const roleName = role.charAt(0).toUpperCase() + role.slice(1);

                item.innerHTML = `
                    <div class="config-label">${roleName} Parallelism</div>
                    <div class="config-value">${parallelStr}</div>
                `;
                configGrid.appendChild(item);
            }
        }
    }

    handleTimeZoom(e, canvas) {
        const rect = canvas.getBoundingClientRect();
        const mouseX = (e.clientX - rect.left) / rect.width;

        const zoom = this.timeZoom;
        const currentRange = zoom.xMax - zoom.xMin;

        if (e.ctrlKey || e.metaKey || Math.abs(e.deltaY) > Math.abs(e.deltaX)) {
            // Zoom: ctrl+wheel or vertical scroll
            const zoomFactor = e.deltaY > 0 ? 1.15 : 0.87;
            const newRange = Math.min(1, Math.max(0.01, currentRange * zoomFactor));

            // Zoom centered on mouse position
            const mouseDataX = zoom.xMin + mouseX * currentRange;
            const newMin = mouseDataX - mouseX * newRange;
            const newMax = mouseDataX + (1 - mouseX) * newRange;

            // Clamp to [0, 1]
            if (newMin < 0) {
                zoom.xMin = 0;
                zoom.xMax = Math.min(1, newRange);
            } else if (newMax > 1) {
                zoom.xMax = 1;
                zoom.xMin = Math.max(0, 1 - newRange);
            } else {
                zoom.xMin = newMin;
                zoom.xMax = newMax;
            }
        } else {
            // Pan: horizontal scroll
            const panAmount = (e.deltaX / rect.width) * currentRange * 0.5;
            const newMin = zoom.xMin + panAmount;
            const newMax = zoom.xMax + panAmount;

            if (newMin >= 0 && newMax <= 1) {
                zoom.xMin = newMin;
                zoom.xMax = newMax;
            } else if (newMin < 0) {
                zoom.xMin = 0;
                zoom.xMax = currentRange;
            } else {
                zoom.xMax = 1;
                zoom.xMin = 1 - currentRange;
            }
        }

        // Render both timeline and buffer (they share time axis)
        this.renderTimeline();
        this.renderBuffer();
    }

    computeUtilization() {
        // Compute utilization from first generation span (excludes startup overhead)
        if (!this.vllmSpans.length) return { vllm: 0, trainer: 0, vllmPerReplica: {}, ddpDegree: 1, startTs: 0 };

        // Find first vLLM generation as start time
        const startTs = Math.min(...this.vllmSpans.map(s => s.ts));
        const duration = this.maxTs - startTs;
        if (duration <= 0) return { vllm: 0, trainer: 0, vllmPerReplica: {}, ddpDegree: 1, startTs };

        // Helper: compute "covered time" (union of potentially overlapping intervals)
        const computeCoveredTime = (spans) => {
            if (!spans.length) return 0;
            // Sort by start time
            const sorted = [...spans].sort((a, b) => a.ts - b.ts);
            let covered = 0;
            let currentEnd = -Infinity;
            for (const s of sorted) {
                const spanStart = s.ts;
                const spanEnd = s.ts + s.dur;
                if (spanStart >= currentEnd) {
                    // No overlap, add the full span
                    covered += s.dur;
                    currentEnd = spanEnd;
                } else if (spanEnd > currentEnd) {
                    // Overlaps but extends past current end
                    covered += spanEnd - currentEnd;
                    currentEnd = spanEnd;
                }
                // else: fully contained in current interval, contributes nothing
            }
            return covered;
        };

        // vLLM: compute covered time per replica, then average
        const replicas = new Set(this.vllmSpans.map(s => s.replica));
        const numVllmReplicas = replicas.size || 1;

        // Per-replica utilization (using covered time, not sum)
        const vllmPerReplica = {};
        let totalCoveredTime = 0;
        for (const r of replicas) {
            const replicaSpans = this.vllmSpans.filter(s => s.replica === r);
            const coveredTime = computeCoveredTime(replicaSpans);
            vllmPerReplica[r] = (coveredTime / duration) * 100;
            totalCoveredTime += coveredTime;
        }
        // Average utilization across replicas
        const vllmUtil = (totalCoveredTime / (duration * numVllmReplicas)) * 100;

        // Detect DDP degree by checking for overlapping trainer spans
        // If spans overlap, we have multiple replicas running in parallel
        let ddpDegree = 1;
        if (this.trainerSpans.length >= 2) {
            // Check first few spans for overlap
            const sorted = [...this.trainerSpans].sort((a, b) => a.ts - b.ts);
            let maxOverlap = 1;
            for (let i = 0; i < Math.min(10, sorted.length); i++) {
                const span = sorted[i];
                const spanEnd = span.ts + span.dur;
                // Count how many spans overlap with this one
                let overlapping = 0;
                for (const other of sorted) {
                    if (other.ts < spanEnd && (other.ts + other.dur) > span.ts) {
                        overlapping++;
                    }
                }
                maxOverlap = Math.max(maxOverlap, overlapping);
            }
            ddpDegree = maxOverlap;
        }

        // Trainer: sum of forward_backward time / (duration * ddp_degree)
        // With DDP, each replica does 1/N of the work, so total GPU-time = sum / ddp_degree
        const trainerTotalTime = this.trainerSpans.reduce((sum, s) => sum + s.dur, 0);
        const trainerUtil = (trainerTotalTime / (duration * ddpDegree)) * 100;

        return { vllm: vllmUtil, trainer: trainerUtil, vllmPerReplica, numVllmReplicas, ddpDegree, startTs };
    }

    computeFateSummary() {
        if (!this.bufferEvents.length) return null;
        const lastEvt = this.bufferEvents[this.bufferEvents.length - 1];
        if (!lastEvt || !lastEvt.fates) return null;

        const { used, wasted, partial } = lastEvt.fates;
        const totalUsed = Object.values(used || {}).reduce((s, v) => s + v, 0);
        const totalWasted = Object.values(wasted || {}).reduce((s, v) => s + v, 0);
        const totalPartial = Object.values(partial || {}).reduce((s, v) => s + v, 0);
        const grandTotal = totalUsed + totalWasted + totalPartial;

        if (grandTotal === 0) return null;

        return {
            usedPct: (totalUsed / grandTotal) * 100,
            wastedPct: (totalWasted / grandTotal) * 100,
            partialPct: (totalPartial / grandTotal) * 100,
            total: grandTotal
        };
    }

    getRecommendations() {
        const util = this.computeUtilization();
        const fates = this.computeFateSummary();
        const recommendations = [];

        if (!fates || fates.total < 10) {
            return [{ type: 'info', text: 'Collecting data...' }];
        }

        // Analyze sample efficiency
        if (fates.wastedPct > 30) {
            recommendations.push({
                type: 'warning',
                text: `${fates.wastedPct.toFixed(0)}% samples wasted - generating too fast`,
                suggestion: 'Increase batch size, reduce vLLM replicas, allow for more staleness'
            });
        } else if (fates.wastedPct > 15) {
            recommendations.push({
                type: 'caution',
                text: `${fates.wastedPct.toFixed(0)}% samples wasted`,
                suggestion: 'Consider increasing batch size or sync interval'
            });
        }

        // Check for vLLM underutilization with good sample efficiency
        if (util.vllm < 50 && fates.wastedPct < 10) {
            recommendations.push({
                type: 'info',
                text: `vLLM ${util.vllm.toFixed(0)}% utilized with low waste`,
                suggestion: 'Could add more vLLM replicas for faster throughput'
            });
        }

        // Check for trainer being the bottleneck
        if (util.trainer > 80 && fates.wastedPct > 20) {
            recommendations.push({
                type: 'warning',
                text: `Trainer busy (${util.trainer.toFixed(0)}%) but samples wasted`,
                suggestion: 'Training throughput is the bottleneck - consider TP or larger batch'
            });
        }

        // Check for trainer starved (low utilization, low waste)
        if (util.trainer < 40 && fates.wastedPct < 5 && util.vllm > 70) {
            recommendations.push({
                type: 'info',
                text: `Trainer ${util.trainer.toFixed(0)}% utilized, waiting for samples`,
                suggestion: 'Trainer may be starved - add more vLLM capacity'
            });
        }

        // Good state
        if (fates.usedPct > 80 && util.trainer > 50 && util.vllm > 50) {
            recommendations.push({
                type: 'success',
                text: `Well balanced: ${fates.usedPct.toFixed(0)}% samples used`,
                suggestion: null
            });
        }

        if (recommendations.length === 0) {
            recommendations.push({
                type: 'info',
                text: `${fates.usedPct.toFixed(0)}% used, vLLM ${util.vllm.toFixed(0)}%, Trainer ${util.trainer.toFixed(0)}%`,
                suggestion: null
            });
        }

        return recommendations;
    }

    renderRecommendations() {
        const container = document.getElementById('overview');
        if (!container) return;

        const recs = this.getRecommendations();
        const util = this.computeUtilization();

        // Build HTML
        let html = '<div class="rec-utilization">';
        html += `<span class="rec-stat">vLLM (${util.numVllmReplicas || 1}x): <strong>${util.vllm.toFixed(0)}%</strong></span>`;
        const ddpLabel = util.ddpDegree > 1 ? ` (DDP ${util.ddpDegree}x)` : '';
        html += `<span class="rec-stat">Trainer${ddpLabel}: <strong>${util.trainer.toFixed(0)}%</strong></span>`;
        html += '</div>';

        html += '<div class="rec-items">';
        for (const rec of recs) {
            const icon = rec.type === 'success' ? '✓' : rec.type === 'warning' ? '!' : rec.type === 'caution' ? '~' : '→';
            html += `<div class="rec-item rec-${rec.type}">`;
            html += `<span class="rec-icon">${icon}</span>`;
            html += `<span class="rec-text">${rec.text}</span>`;
            if (rec.suggestion) {
                html += `<span class="rec-suggestion">${rec.suggestion}</span>`;
            }
            html += '</div>';
        }
        html += '</div>';

        container.innerHTML = html;
    }

    renderTimeline() {
        const canvas = document.getElementById('timeline-canvas');
        const ctx = this.timelineCtx;

        // Handle high DPI - use canvas's own rect, let CSS control size
        const rect = canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        const width = rect.width;
        const height = rect.height;

        if (width <= 0 || height <= 0) return;

        canvas.width = Math.floor(width * dpr);
        canvas.height = Math.floor(height * dpr);
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        // Clear
        ctx.fillStyle = '#0d1117';
        ctx.fillRect(0, 0, width, height);

        if (!this.events.length) {
            ctx.fillStyle = '#8b949e';
            ctx.font = '14px -apple-system, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Load a trace file to visualize', width / 2, height / 2);
            return;
        }

        const duration = this.maxTs;  // Already in seconds
        const padding = { left: 100, right: 20, top: 20, bottom: 30 };
        const plotWidth = width - padding.left - padding.right;
        const plotHeight = height - padding.top - padding.bottom;

        // Apply zoom
        const zoom = this.timeZoom;
        const zoomRange = zoom.xMax - zoom.xMin;
        const visibleStart = duration * zoom.xMin;
        const visibleEnd = duration * zoom.xMax;
        const visibleDuration = visibleEnd - visibleStart;

        // Helper to convert time to x coordinate
        const timeToX = (t) => {
            return padding.left + ((t - visibleStart) / visibleDuration) * plotWidth;
        };

        // Draw zoom indicator if zoomed
        if (zoomRange < 0.99) {
            ctx.fillStyle = '#58a6ff';
            ctx.font = '10px SF Mono, Monaco, monospace';
            ctx.textAlign = 'right';
            ctx.textBaseline = 'top';
            const pct = Math.round((1 / zoomRange) * 100);
            ctx.fillText(`${pct}%`, width - 5, 5);
        }

        // Build dynamic swimlanes based on actual data
        const lanes = [];

        // vLLM lanes (dynamic count, max 8)
        // Use vllm.generate_single which has replica info, or vllm.generate/generate if they have replica
        const numVllm = Math.min(8, Math.max(1, this.numVllmReplicas));
        const isVllmSpan = (e) => {
            if (e.name === 'vllm.generate_single') return true;
            if (e.name === 'vllm.generate') return true;
            // Only include 'generate' if it has replica info (otherwise it's the outer wrapper)
            if (e.name === 'generate' && e.replica !== undefined) return true;
            return false;
        };
        for (let i = 0; i < numVllm; i++) {
            lanes.push({
                name: numVllm === 1 ? 'vLLM' : `vLLM ${i}`,
                filter: e => e.type === 'span' && isVllmSpan(e) && (e.replica || 0) === i
            });
        }

        // Fixed lanes
        lanes.push({ name: 'Reference', filter: e => e.type === 'span' && e.name === 'ref_logprobs' });
        lanes.push({ name: 'Trainer', filter: e => e.type === 'span' && e.name === 'forward_backward' });
        lanes.push({ name: 'Sync', filter: e => e.type === 'span' && e.name.startsWith('sync.') });

        // Verifier lanes (workers × max_concurrent slots) - only if not collapsed
        if (!this.verifiersCollapsed) {
            const numVerifierWorkers = this.numVerifierWorkers;
            const maxConcurrent = Math.max(1, this.verifierMaxConcurrent);
            for (let w = 0; w < numVerifierWorkers; w++) {
                for (let s = 0; s < maxConcurrent; s++) {
                    lanes.push({
                        name: `V${w}-${s}`,  // Compact: "V0-0", "V0-1", etc.
                        filter: e => e.type === 'span' && e.name === 'verify' && e.worker === w && (e.slot === s || (e.slot === undefined && s === 0))
                    });
                }
            }
        }

        const laneHeight = plotHeight / lanes.length;

        // Draw lane labels and lines
        ctx.fillStyle = '#8b949e';
        ctx.font = '11px -apple-system, sans-serif';
        ctx.textAlign = 'right';

        lanes.forEach((lane, i) => {
            const y = padding.top + i * laneHeight + laneHeight / 2;
            ctx.fillText(lane.name, padding.left - 10, y + 4);

            // Lane separator
            ctx.strokeStyle = '#21262d';
            ctx.beginPath();
            ctx.moveTo(padding.left, padding.top + (i + 1) * laneHeight);
            ctx.lineTo(width - padding.right, padding.top + (i + 1) * laneHeight);
            ctx.stroke();
        });

        // Draw events
        const colors = {
            'vllm.generate': '#3fb950',
            'vllm.generate_single': '#3fb950',
            'generate': '#3fb950',
            'forward_backward': '#58a6ff',
            'ref_logprobs': '#a371f7',
            'sync.trainer_to_vllm': '#f85149',
            'sync.trainer_to_reference': '#f0883e',
            'sync.waiting_for_vllm_pause': '#f8514966',
            'sync.waiting_for_dst_idle': '#f0883e66',
            'sync.titan_to_vllm': '#f85149',
            'sync.titan_to_titan': '#f0883e',
            'verify': '#39d4e0'  // Cyan for verifier
        };

        // Clear span geometries for hit testing
        this.timelineSpans = [];

        for (const event of this.events) {
            if (event.type !== 'span') continue;

            const laneIdx = lanes.findIndex(l => l.filter(event));
            if (laneIdx === -1) continue;

            const startTs = event.ts;
            const endTs = startTs + (event.dur || 0);

            // Skip events outside visible range
            if (endTs < visibleStart || startTs > visibleEnd) continue;

            const x = timeToX(startTs);
            const xEnd = timeToX(endTs);
            const w = Math.max(1, xEnd - x);
            const y = padding.top + laneIdx * laneHeight + 4;
            const h = laneHeight - 8;

            const color = colors[event.name] || '#8b949e';

            // For verify spans: solid fill if passed, outline only if failed
            if (event.name === 'verify' && event.passed === 0) {
                // Draw outline only for failed verifications
                ctx.strokeStyle = color;
                ctx.lineWidth = 1;
                ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
            } else {
                // Solid fill for passed verifications and other events
                ctx.fillStyle = color;
                ctx.fillRect(x, y, w, h);
            }

            // Store geometry for hit testing
            this.timelineSpans.push({ event, x, y, w, h });
        }

        // Draw time axis
        ctx.fillStyle = '#8b949e';
        ctx.font = '10px -apple-system, sans-serif';
        ctx.textAlign = 'center';

        const numTicks = 10;
        for (let i = 0; i <= numTicks; i++) {
            const t = visibleStart + (visibleDuration * i / numTicks);
            const x = padding.left + (i / numTicks) * plotWidth;
            ctx.fillText(t.toFixed(1) + 's', x, height - 10);
        }
    }

    renderBuffer() {
        const canvas = document.getElementById('buffer-canvas');
        const ctx = this.bufferCtx;

        // Handle high DPI
        const rect = canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        const width = rect.width;
        const height = rect.height;

        if (width <= 0 || height <= 0) return;

        canvas.width = Math.floor(width * dpr);
        canvas.height = Math.floor(height * dpr);
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        // Clear
        ctx.fillStyle = '#0d1117';
        ctx.fillRect(0, 0, width, height);

        if (!this.bufferEvents.length) {
            ctx.fillStyle = '#8b949e';
            ctx.font = '14px -apple-system, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Buffer dynamics will appear here', width / 2, height / 2);
            return;
        }

        const duration = this.maxTs;  // Already in seconds
        const padding = { left: 50, right: 20, top: 20, bottom: 30 };
        const plotWidth = width - padding.left - padding.right;
        const plotHeight = height - padding.top - padding.bottom;

        // Apply zoom (synced with timeline)
        const zoom = this.timeZoom;
        const zoomRange = zoom.xMax - zoom.xMin;
        const visibleStart = duration * zoom.xMin;
        const visibleEnd = duration * zoom.xMax;
        const visibleDuration = visibleEnd - visibleStart;

        // Helper to convert time to x coordinate
        const timeToX = (t) => {
            return padding.left + ((t - visibleStart) / visibleDuration) * plotWidth;
        };

        // Draw zoom indicator if zoomed
        if (zoomRange < 0.99) {
            ctx.fillStyle = '#58a6ff';
            ctx.font = '10px SF Mono, Monaco, monospace';
            ctx.textAlign = 'right';
            ctx.textBaseline = 'top';
            const pct = Math.round((1 / zoomRange) * 100);
            ctx.fillText(`${pct}%`, width - 5, 5);
        }

        // Version colors (cycle through 10 distinct colors)
        const versionColors = [
            '#3fb950', '#58a6ff', '#a371f7', '#f0883e', '#39d4e0',
            '#d29922', '#f85149', '#8b949e', '#2ea88a', '#bc8cff'
        ];

        // Collect all versions that appear
        const allVersions = new Set();
        for (const evt of this.bufferEvents) {
            for (const v of Object.keys(evt.byVersion || {})) {
                allVersions.add(parseInt(v));
            }
        }
        const versions = [...allVersions].sort((a, b) => a - b);

        // Find max size for scaling
        let maxSize = Math.max(...this.bufferEvents.map(e => e.size), 1);

        // Draw proper stacked area chart using actual by_version data
        // Each version stacks on top of previous versions, showing true overlap
        //
        // Buffer events are now emitted on every state change (put/pop/evict),
        // so we just draw what we see - no synthetic points needed.

        if (versions.length > 0 && this.bufferEvents.length > 0) {
            // Build stacked data directly from buffer events
            const stackedData = this.bufferEvents.map(evt => {
                const result = [];
                let cumulative = 0;
                for (const version of versions) {
                    const count = evt.byVersion[version] || 0;
                    result.push({ bottom: cumulative, top: cumulative + count, count });
                    cumulative += count;
                }
                return result;
            });

            // Draw from bottom to top (oldest version first)
            for (let vi = 0; vi < versions.length; vi++) {
                const version = versions[vi];
                const color = versionColors[version % versionColors.length];

                // Build path points for this version's band (only visible range)
                const points = [];
                for (let i = 0; i < this.bufferEvents.length; i++) {
                    const evt = this.bufferEvents[i];
                    // Skip events outside visible range (with some margin for continuity)
                    if (evt.ts < visibleStart && i < this.bufferEvents.length - 1 && this.bufferEvents[i + 1].ts < visibleStart) continue;
                    if (evt.ts > visibleEnd && points.length > 0) break;

                    const stack = stackedData[i][vi];
                    points.push({
                        ts: evt.ts,
                        x: timeToX(evt.ts),
                        bottom: height - padding.bottom - (stack.bottom / maxSize) * plotHeight,
                        top: height - padding.bottom - (stack.top / maxSize) * plotHeight,
                        count: stack.count
                    });
                }

                // Draw the filled band (top edge forward, bottom edge backward)
                ctx.beginPath();
                ctx.moveTo(points[0].x, points[0].bottom);

                // Draw top edge (forward) - use step function (horizontal then vertical)
                for (let i = 0; i < points.length; i++) {
                    const p = points[i];
                    if (i > 0) {
                        // Step: horizontal to new x at old y, then vertical to new y
                        const prev = points[i - 1];
                        ctx.lineTo(p.x, prev.top);  // horizontal
                    }
                    ctx.lineTo(p.x, p.top);  // vertical (or first point)
                }

                // Draw bottom edge (backward) - also step function
                for (let i = points.length - 1; i >= 0; i--) {
                    const p = points[i];
                    if (i < points.length - 1) {
                        const next = points[i + 1];
                        ctx.lineTo(p.x, next.bottom);  // horizontal
                    }
                    ctx.lineTo(p.x, p.bottom);  // vertical
                }

                ctx.closePath();
                ctx.fillStyle = color + 'cc';
                ctx.fill();

                // Draw top edge outline
                ctx.beginPath();
                for (let i = 0; i < points.length; i++) {
                    const p = points[i];
                    if (i === 0) {
                        ctx.moveTo(p.x, p.top);
                    } else {
                        const prev = points[i - 1];
                        ctx.lineTo(p.x, prev.top);  // horizontal
                        ctx.lineTo(p.x, p.top);     // vertical
                    }
                }
                ctx.strokeStyle = color;
                ctx.lineWidth = 1;
                ctx.stroke();
            }
        } else {
            // Fallback: simple line chart if no version data
            ctx.beginPath();
            ctx.moveTo(padding.left, height - padding.bottom);
            for (const point of this.bufferEvents) {
                const x = padding.left + (point.ts / duration) * plotWidth;
                const y = height - padding.bottom - (point.size / maxSize) * plotHeight;
                ctx.lineTo(x, y);
            }
            const lastX = padding.left + (this.bufferEvents[this.bufferEvents.length - 1].ts / duration) * plotWidth;
            ctx.lineTo(lastX, height - padding.bottom);
            ctx.closePath();
            ctx.fillStyle = '#a371f766';
            ctx.fill();
        }

        // Draw sync event vertical lines (only visible ones)
        for (const sync of this.syncEvents) {
            if (sync.ts < visibleStart || sync.ts > visibleEnd) continue;
            const x = timeToX(sync.ts);
            ctx.strokeStyle = '#f8514966';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(x, padding.top);
            ctx.lineTo(x, height - padding.bottom);
            ctx.stroke();

            // Version label
            ctx.fillStyle = '#f85149';
            ctx.font = '10px -apple-system, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('v' + sync.version, x, padding.top - 5);
        }

        // Draw legend for versions (compact, bottom right)
        if (versions.length > 0) {
            const legendX = width - padding.right - 10;
            const legendY = height - padding.bottom - 10;
            ctx.font = '9px -apple-system, sans-serif';
            ctx.textAlign = 'right';
            for (let i = Math.min(versions.length - 1, 4); i >= 0; i--) {
                const v = versions[versions.length - 1 - (4 - i)];
                if (v === undefined) continue;
                const color = versionColors[v % versionColors.length];
                const y = legendY - i * 12;
                ctx.fillStyle = color;
                ctx.fillRect(legendX - 30, y - 8, 8, 8);
                ctx.fillStyle = '#8b949e';
                ctx.fillText('v' + v, legendX, y);
            }
        }

        // Y axis labels
        ctx.fillStyle = '#8b949e';
        ctx.font = '10px -apple-system, sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText('0', padding.left - 5, height - padding.bottom);
        ctx.fillText(maxSize.toString(), padding.left - 5, padding.top + 10);

        // Time axis
        ctx.textAlign = 'center';
        const numTicks = 10;
        for (let i = 0; i <= numTicks; i++) {
            const t = visibleStart + (visibleDuration * i / numTicks);
            const x = padding.left + (i / numTicks) * plotWidth;
            ctx.fillText(t.toFixed(1) + 's', x, height - 10);
        }
    }

    renderFates() {
        const canvas = document.getElementById('fates-canvas');
        const ctx = this.fatesCtx;

        // Handle high DPI
        const rect = canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        const width = rect.width;
        const height = rect.height;

        if (width <= 0 || height <= 0) return;

        canvas.width = Math.floor(width * dpr);
        canvas.height = Math.floor(height * dpr);
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        // Clear
        ctx.fillStyle = '#0d1117';
        ctx.fillRect(0, 0, width, height);

        if (!this.bufferEvents.length) {
            ctx.fillStyle = '#8b949e';
            ctx.font = '14px -apple-system, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Sample fates will appear here', width / 2, height / 2);
            return;
        }

        // Get latest fates from last buffer event
        const lastEvt = this.bufferEvents[this.bufferEvents.length - 1];
        if (!lastEvt || !lastEvt.fates) {
            ctx.fillStyle = '#8b949e';
            ctx.font = '12px -apple-system, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('No fate data yet', width / 2, height / 2);
            return;
        }

        const { used, wasted, partial, filtered } = lastEvt.fates;

        // Collect all versions from fates
        const allVersions = new Set([
            ...Object.keys(used || {}),
            ...Object.keys(wasted || {}),
            ...Object.keys(partial || {}),
            ...Object.keys(filtered || {})
        ].map(v => parseInt(v)));
        const versions = [...allVersions].sort((a, b) => a - b);

        // Calculate totals per version
        const fateData = [];
        for (const v of versions) {
            const u = (used || {})[v] || 0;
            const w = (wasted || {})[v] || 0;
            const p = (partial || {})[v] || 0;
            const f = (filtered || {})[v] || 0;
            const total = u + w + p + f;
            if (total > 0) {
                fateData.push({ version: v, used: u, wasted: w, partial: p, filtered: f, total });
            }
        }

        if (fateData.length === 0) {
            ctx.fillStyle = '#8b949e';
            ctx.font = '12px -apple-system, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('No completed samples yet', width / 2, height / 2);
            return;
        }

        const padding = { left: 35, right: 50, top: 10, bottom: 20 };
        const plotWidth = width - padding.left - padding.right;
        const plotHeight = height - padding.top - padding.bottom;

        // Calculate how many versions fit on screen
        const maxVisible = 8;
        const totalVersions = fateData.length;

        // Apply scroll offset (clamp to valid range)
        const maxOffset = Math.max(0, totalVersions - maxVisible);
        this.fatesScrollOffset = Math.min(this.fatesScrollOffset, maxOffset);

        // Get visible slice based on scroll
        const startIdx = Math.max(0, totalVersions - maxVisible - this.fatesScrollOffset);
        const endIdx = totalVersions - this.fatesScrollOffset;
        const recentFates = fateData.slice(startIdx, endIdx);

        // Draw scroll indicator if there are more versions
        if (totalVersions > maxVisible) {
            ctx.fillStyle = '#58a6ff';
            ctx.font = '9px SF Mono, Monaco, monospace';
            ctx.textAlign = 'right';
            ctx.textBaseline = 'top';
            const scrollInfo = this.fatesScrollOffset > 0 ? `↑${this.fatesScrollOffset} more` : `${totalVersions} versions`;
            ctx.fillText(scrollInfo, width - 5, 2);
            if (startIdx > 0) {
                ctx.textBaseline = 'bottom';
                ctx.fillText(`↓${startIdx} more`, width - 5, height - 2);
            }
        }

        const barHeight = Math.min(24, (plotHeight - 10) / recentFates.length - 4);
        const barSpacing = (plotHeight - recentFates.length * barHeight) / (recentFates.length + 1);

        // Find max total for scaling
        const maxTotal = Math.max(...recentFates.map(f => f.total));

        // Clear fate bars for hover detection
        this.fateBars = [];

        for (let i = 0; i < recentFates.length; i++) {
            const f = recentFates[i];
            const y = padding.top + barSpacing + i * (barHeight + barSpacing);

            // Version label
            ctx.fillStyle = '#8b949e';
            ctx.font = '11px -apple-system, sans-serif';
            ctx.textAlign = 'right';
            ctx.fillText('v' + f.version, padding.left - 5, y + barHeight / 2 + 4);

            // Calculate bar width proportional to total
            const totalBarWidth = (f.total / maxTotal) * plotWidth;

            // Store bar position for hover detection
            this.fateBars.push({
                x: padding.left,
                y: y,
                w: totalBarWidth,
                h: barHeight,
                fate: f
            });

            // Stacked bar
            let x = padding.left;
            const scale = totalBarWidth / f.total;

            // Used (green)
            if (f.used > 0) {
                ctx.fillStyle = '#3fb950';
                const w = f.used * scale;
                ctx.fillRect(x, y, w, barHeight);
                x += w;
            }
            // Partial (yellow)
            if (f.partial > 0) {
                ctx.fillStyle = '#d29922';
                const w = f.partial * scale;
                ctx.fillRect(x, y, w, barHeight);
                x += w;
            }
            // Wasted (red)
            if (f.wasted > 0) {
                ctx.fillStyle = '#f85149';
                const w = f.wasted * scale;
                ctx.fillRect(x, y, w, barHeight);
                x += w;
            }
            // Filtered (gray/blue)
            if (f.filtered > 0) {
                ctx.fillStyle = '#8b949e';
                const w = f.filtered * scale;
                ctx.fillRect(x, y, w, barHeight);
            }

            // Total count only (no percentage - hover for details)
            ctx.fillStyle = '#c9d1d9';
            ctx.font = '10px -apple-system, sans-serif';
            ctx.textAlign = 'left';
            ctx.fillText(`${f.total}`, padding.left + totalBarWidth + 5, y + barHeight / 2 + 4);
        }

        // Summary at bottom
        const totalUsed = fateData.reduce((s, f) => s + f.used, 0);
        const totalWasted = fateData.reduce((s, f) => s + f.wasted, 0);
        const totalPartial = fateData.reduce((s, f) => s + f.partial, 0);
        const totalFiltered = fateData.reduce((s, f) => s + f.filtered, 0);
        const grandTotal = totalUsed + totalWasted + totalPartial + totalFiltered;

        if (grandTotal > 0) {
            ctx.fillStyle = '#8b949e';
            ctx.font = '10px -apple-system, sans-serif';
            ctx.textAlign = 'center';
            const usedPct = Math.round(100 * totalUsed / grandTotal);
            const filteredPct = Math.round(100 * totalFiltered / grandTotal);
            ctx.fillText(
                `Total: ${grandTotal} | ${usedPct}% used, ${filteredPct}% filtered`,
                width / 2,
                height - 5
            );
        }
    }

    renderMetrics() {
        for (const metric of Object.keys(this.metrics)) {
            this.renderMetricChart(metric);
        }
    }

    renderEfficiencyQuantiles() {
        // Render quantile chart for completion lengths
        const complTotal = this.histograms.completion_lens.reduce((a, b) => a + b, 0);
        console.log('[renderEfficiencyQuantiles] histogram total:', complTotal, 'bins with data:', this.histograms.completion_lens.filter(x => x > 0).length);
        this.renderQuantileChart('completion_len_mean', this.histograms.completion_lens, this.histogramBinSize, '', 'completion_len_mean');

        // Render quantile charts for padding token counts (absolute waste)
        this.renderQuantileChart('seq_padding_pct', this.histograms.seq_padding_tokens, this.paddingBinSize, ' toks', 'seq_padding_tokens');
        this.renderQuantileChart('completion_padding_pct', this.histograms.completion_padding_tokens, this.paddingBinSize, ' toks', 'completion_padding_tokens');
    }

    /**
     * Render a quantile/CDF chart from a histogram.
     * X-axis: percentile (0-100), Y-axis: value
     * @param {string} metricName - Name of the metric (for canvas lookup and display)
     * @param {number[]} histogram - Array of bin counts
     * @param {number} binSize - Size of each bin (default: this.histogramBinSize)
     * @param {string} unit - Unit suffix for display (default: '' for tokens, '%' for percentages)
     * @param {string} hoverKey - Key in quantileHover for hover state (default: metricName)
     */
    renderQuantileChart(metricName, histogram, binSize = this.histogramBinSize, unit = '', hoverKey = null) {
        hoverKey = hoverKey || metricName;
        const ctx = this.metricCharts[metricName];
        if (!ctx) return;

        const canvas = ctx.canvas;
        const rect = canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        const width = rect.width;
        const height = rect.height;

        if (width <= 0 || height <= 0) return;

        const targetWidth = Math.floor(width * dpr);
        const targetHeight = Math.floor(height * dpr);

        if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
            canvas.width = targetWidth;
            canvas.height = targetHeight;
        }

        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        const padding = { left: 35, right: 5, top: 3, bottom: 3 };
        const plotWidth = width - padding.left - padding.right;
        const plotHeight = height - padding.top - padding.bottom;

        ctx.clearRect(0, 0, width, height);

        const total = histogram.reduce((a, b) => a + b, 0);
        if (total === 0) return;

        // Build CDF: for each percentile (0-100), what's the value?
        // We'll sample at 100 points for smooth curve
        const cdfPoints = [];
        let cumulative = 0;
        let binIdx = 0;
        for (let p = 0; p <= 100; p++) {
            const target = (p / 100) * total;
            while (binIdx < histogram.length && cumulative + histogram[binIdx] < target) {
                cumulative += histogram[binIdx];
                binIdx++;
            }
            // Interpolate within bin
            const binValue = (binIdx + 0.5) * binSize;
            cdfPoints.push({ p, value: binValue });
        }

        // Find max value for y-axis scaling
        const maxValue = Math.max(...cdfPoints.map(pt => pt.value), binSize);

        // Update the metric value display with p50/p90/p99
        const [p50, p90, p99] = this.computeQuantilesWithBinSize(histogram, [0.5, 0.9, 0.99], binSize);
        const el = document.getElementById(`metric-${metricName}`);
        if (el) {
            const fmt = (v) => unit === '%' ? v.toFixed(1) + unit : Math.round(v).toString();
            el.innerHTML = `<span style="color: #8b949e">${fmt(p50)}</span> / ` +
                           `<span style="color: #d29922">${fmt(p90)}</span> / ` +
                           `<span style="color: #f85149">${fmt(p99)}</span>`;
        }

        // Draw y-axis labels
        ctx.fillStyle = '#6e7681';
        ctx.font = '9px SF Mono, Monaco, monospace';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'top';
        const maxLabel = unit === '%' ? maxValue.toFixed(0) + unit : Math.round(maxValue).toString();
        ctx.fillText(maxLabel, padding.left - 4, padding.top);
        ctx.textBaseline = 'bottom';
        ctx.fillText('0' + unit, padding.left - 4, height - padding.bottom);

        // Draw subtle axis line
        ctx.strokeStyle = '#30363d';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(padding.left, padding.top);
        ctx.lineTo(padding.left, height - padding.bottom);
        ctx.stroke();

        // Draw the CDF curve (percentile on x, value on y)
        ctx.beginPath();
        for (let i = 0; i < cdfPoints.length; i++) {
            const pt = cdfPoints[i];
            const x = padding.left + (pt.p / 100) * plotWidth;
            const y = height - padding.bottom - (pt.value / maxValue) * plotHeight;
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        ctx.strokeStyle = '#58a6ff';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Draw p50, p90, p99 markers
        const markers = [
            { p: 50, value: p50, color: '#8b949e' },
            { p: 90, value: p90, color: '#d29922' },
            { p: 99, value: p99, color: '#f85149' }
        ];
        for (const m of markers) {
            const x = padding.left + (m.p / 100) * plotWidth;
            const y = height - padding.bottom - (m.value / maxValue) * plotHeight;
            ctx.fillStyle = m.color;
            ctx.beginPath();
            ctx.arc(x, y, 2.5, 0, Math.PI * 2);
            ctx.fill();
        }

        // Draw hover indicator
        const hoverPct = this.quantileHover[hoverKey];
        if (hoverPct !== null && hoverPct >= 0 && hoverPct <= 1) {
            const pctIdx = Math.round(hoverPct * 100);
            const pt = cdfPoints[Math.min(pctIdx, cdfPoints.length - 1)];
            const x = padding.left + hoverPct * plotWidth;
            const y = height - padding.bottom - (pt.value / maxValue) * plotHeight;

            // Vertical line
            ctx.strokeStyle = '#6e7681';
            ctx.lineWidth = 1;
            ctx.setLineDash([2, 2]);
            ctx.beginPath();
            ctx.moveTo(x, padding.top);
            ctx.lineTo(x, height - padding.bottom);
            ctx.stroke();
            ctx.setLineDash([]);

            // Point
            ctx.fillStyle = '#58a6ff';
            ctx.beginPath();
            ctx.arc(x, y, 3, 0, Math.PI * 2);
            ctx.fill();

            // Value label with background
            const valStr = unit === '%' ? pt.value.toFixed(1) + unit : Math.round(pt.value).toString();
            const label = `p${pt.p}: ${valStr}`;
            ctx.font = '10px SF Mono, Monaco, monospace';
            const textWidth = ctx.measureText(label).width;
            const labelX = x > width / 2 ? x - textWidth - 8 : x + 6;

            ctx.fillStyle = '#161b22';
            ctx.fillRect(labelX - 2, 2, textWidth + 4, 12);

            ctx.fillStyle = '#f0f6fc';
            ctx.textAlign = 'left';
            ctx.textBaseline = 'top';
            ctx.fillText(label, labelX, 3);
        }
    }

    /**
     * Compute quantiles from a binned histogram with custom bin size.
     * @param {number[]} histogram - Array of bin counts
     * @param {number[]} quantiles - Array of quantiles to compute (e.g., [0.5, 0.9, 0.99])
     * @param {number} binSize - Size of each bin
     * @returns {number[]} - Array of values at each quantile (bin midpoints)
     */
    computeQuantilesWithBinSize(histogram, quantiles, binSize) {
        const total = histogram.reduce((a, b) => a + b, 0);
        if (total === 0) return quantiles.map(() => 0);

        const results = [];
        for (const q of quantiles) {
            const target = q * total;
            let cumulative = 0;
            for (let i = 0; i < histogram.length; i++) {
                cumulative += histogram[i];
                if (cumulative >= target) {
                    // Return bin midpoint
                    results.push((i + 0.5) * binSize);
                    break;
                }
            }
            if (results.length < quantiles.indexOf(q) + 1) {
                // Fell through, use last bin
                results.push((histogram.length - 0.5) * binSize);
            }
        }
        return results;
    }

    renderMetricChart(metric) {
        // Skip metrics that use renderQuantileChart instead
        if (metric === 'completion_len_mean' || metric === 'seq_padding_pct' || metric === 'completion_padding_pct') return;

        const colors = {
            reward: '#3fb950',
            loss: '#58a6ff',
            grad_norm: '#f85149',
            mfu: '#a371f7',
            train_tps: '#d29922',
            memory_pct: '#e3b341',
            vllm_tps: '#39d4e0',
            vllm_output_tokens: '#2ea88a',
            vllm_prompt_tokens: '#8b949e',
            kl_max: '#f85149',
            ratio_max: '#d29922',
            diff_ref: '#a371f7',
            diff_rollout: '#39d4e0',
            // Efficiency metrics
            seq_padding_pct: '#f0883e',      // orange - padding is waste
            completion_padding_pct: '#d29922', // yellow
            finish_stop_pct: '#3fb950',       // green - good (natural stop)
            finish_length_pct: '#f85149'      // red - truncated
        };

        const data = this.metrics[metric];
        const ctx = this.metricCharts[metric];
        if (!ctx) return;

        const canvas = ctx.canvas;
        const rect = canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;

        // Use canvas's actual displayed size
        const width = rect.width;
        const height = rect.height;

        if (width <= 0 || height <= 0) return;

        const targetWidth = Math.floor(width * dpr);
        const targetHeight = Math.floor(height * dpr);

        // Only resize canvas buffer if dimensions changed (avoids reflow on hover)
        // Never touch canvas.style - let CSS control that
        if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
            canvas.width = targetWidth;
            canvas.height = targetHeight;
        }

        // Reset transform and apply scale
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        const padding = { left: 35, right: 5, top: 3, bottom: 3 };
        const plotWidth = width - padding.left - padding.right;
        const plotHeight = height - padding.top - padding.bottom;

        // Clear
        ctx.clearRect(0, 0, width, height);

        if (data.length === 0) return;

        // Update current value display
        // Use rolling average for perf metrics (TorchTitan/vLLM), actual value for training metrics
        // Skip completion_len_mean - it's handled by renderEfficiencyQuantiles with p50/90/99
        const perfMetrics = new Set(['mfu', 'train_tps', 'memory_pct', 'vllm_tps', 'vllm_output_tokens', 'vllm_prompt_tokens']);
        const el = document.getElementById(`metric-${metric}`);
        if (el && metric !== 'completion_len_mean') {
            if (perfMetrics.has(metric)) {
                // Rolling average for perf metrics
                const avgWindow = 10;
                const recentValues = data.slice(-avgWindow).map(d => d.value);
                const avgValue = recentValues.reduce((a, b) => a + b, 0) / recentValues.length;
                const formattedValue = this.formatMetricValue(metric, avgValue);
                const windowLabel = recentValues.length < avgWindow ? `avg ${recentValues.length}` : 'avg 10';
                el.innerHTML = `${formattedValue} <span style="font-size: 9px; color: #6e7681;">(${windowLabel})</span>`;
            } else {
                // Actual last value for training/debug metrics
                const lastValue = data[data.length - 1].value;
                el.textContent = this.formatMetricValue(metric, lastValue);
            }
        }

        // Get zoom state
        const zoom = this.metricZoom[metric] || { xMin: 0, xMax: 1 };
        const zoomRange = zoom.xMax - zoom.xMin;

        // Calculate visible data range based on zoom
        const startIdx = Math.floor(zoom.xMin * (data.length - 1));
        const endIdx = Math.ceil(zoom.xMax * (data.length - 1));
        const visibleData = data.slice(startIdx, endIdx + 1);

        if (visibleData.length === 0) return;

        // Determine if log scale
        const useLog = this.logScaleMetrics.has(metric);

        // Get values for visible range, apply log if needed
        const rawValues = visibleData.map(d => d.value);
        const values = useLog ? rawValues.map(v => v > 0 ? Math.log10(v) : -10) : rawValues;

        let minVal = Math.min(...values);
        let maxVal = Math.max(...values);

        // Add some padding to the range
        const rangePadding = (maxVal - minVal) * 0.1 || 0.1;
        minVal -= rangePadding;
        maxVal += rangePadding;
        const range = maxVal - minVal || 1;

        // Helper to convert value to y coordinate
        const valueToY = (val) => {
            const v = useLog ? (val > 0 ? Math.log10(val) : -10) : val;
            return padding.top + plotHeight - ((v - minVal) / range) * plotHeight;
        };

        // Helper to convert data index to x coordinate (accounting for zoom)
        const idxToX = (idx) => {
            const normalized = idx / (data.length - 1 || 1);  // 0-1 in full data
            const visible = (normalized - zoom.xMin) / zoomRange;  // 0-1 in visible range
            return padding.left + visible * plotWidth;
        };

        // Draw y-axis labels (min and max)
        ctx.fillStyle = '#6e7681';
        ctx.font = '9px SF Mono, Monaco, monospace';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'top';

        const maxLabel = useLog ? Math.pow(10, maxVal).toPrecision(2) : this.formatAxisValue(maxVal);
        const minLabel = useLog ? Math.pow(10, minVal).toPrecision(2) : this.formatAxisValue(minVal);

        ctx.fillText(maxLabel, padding.left - 4, padding.top);
        ctx.textBaseline = 'bottom';
        ctx.fillText(minLabel, padding.left - 4, height - padding.bottom);

        // Draw subtle axis line
        ctx.strokeStyle = '#30363d';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(padding.left, padding.top);
        ctx.lineTo(padding.left, height - padding.bottom);
        ctx.stroke();

        // Draw zoom indicator if zoomed
        if (zoomRange < 0.99) {
            ctx.fillStyle = '#58a6ff';
            ctx.font = '8px SF Mono, Monaco, monospace';
            ctx.textAlign = 'right';
            ctx.textBaseline = 'top';
            const pct = Math.round((1 / zoomRange) * 100);
            ctx.fillText(`${pct}%`, width - 2, 1);
        }

        // Draw sparkline
        ctx.beginPath();
        let firstPoint = true;
        for (let i = startIdx; i <= endIdx && i < data.length; i++) {
            const x = idxToX(i);
            const y = valueToY(data[i].value);
            if (firstPoint) {
                ctx.moveTo(x, y);
                firstPoint = false;
            } else {
                ctx.lineTo(x, y);
            }
        }

        ctx.strokeStyle = colors[metric];
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Draw hover indicator and value
        if (this.hoverMetric === metric && data.length > 0) {
            // Map hover position to data index accounting for zoom
            const dataFrac = zoom.xMin + this.hoverX * zoomRange;
            const idx = Math.max(0, Math.min(data.length - 1, Math.round(dataFrac * (data.length - 1))));
            const x = idxToX(idx);
            const y = valueToY(data[idx].value);

            // Vertical line
            ctx.strokeStyle = '#6e7681';
            ctx.lineWidth = 1;
            ctx.setLineDash([2, 2]);
            ctx.beginPath();
            ctx.moveTo(x, padding.top);
            ctx.lineTo(x, height - padding.bottom);
            ctx.stroke();
            ctx.setLineDash([]);

            // Point
            ctx.fillStyle = colors[metric];
            ctx.beginPath();
            ctx.arc(x, y, 3, 0, Math.PI * 2);
            ctx.fill();

            // Value label with background
            const value = this.formatMetricValue(metric, data[idx].value);
            ctx.font = '10px SF Mono, Monaco, monospace';
            const textWidth = ctx.measureText(value).width;
            const labelX = x > width / 2 ? x - textWidth - 8 : x + 6;

            ctx.fillStyle = '#161b22';
            ctx.fillRect(labelX - 2, 2, textWidth + 4, 12);

            ctx.fillStyle = '#f0f6fc';
            ctx.textAlign = 'left';
            ctx.textBaseline = 'top';
            ctx.fillText(value, labelX, 3);
        }
    }

    formatAxisValue(value) {
        const abs = Math.abs(value);
        if (abs >= 1000000) return (value / 1000000).toFixed(1) + 'M';
        if (abs >= 1000) return (value / 1000).toFixed(1) + 'K';
        if (abs >= 100) return value.toFixed(0);
        if (abs >= 10) return value.toFixed(1);
        if (abs >= 1) return value.toFixed(2);
        return value.toPrecision(2);
    }

    formatMetricValue(metric, value) {
        switch (metric) {
            case 'reward':
                return value.toFixed(2);
            case 'loss':
                return value.toFixed(4);
            case 'grad_norm':
                return value.toFixed(2);
            case 'mfu':
            case 'memory_pct':
            case 'seq_padding_pct':
            case 'completion_padding_pct':
            case 'finish_stop_pct':
            case 'finish_length_pct':
                return value.toFixed(1) + '%';
            case 'completion_len_mean':
                return Math.round(value).toLocaleString();
            case 'train_tps':
            case 'vllm_tps':
            case 'vllm_output_tokens':
            case 'vllm_prompt_tokens':
                return Math.round(value).toLocaleString();
            case 'kl_max':
            case 'ratio_max':
            case 'diff_ref':
            case 'diff_rollout':
                return value.toFixed(4);
            default:
                return value.toFixed(2);
        }
    }

    updateTimelineTooltip() {
        let tooltip = document.getElementById('timeline-tooltip');

        if (!this.hoveredSpan) {
            if (tooltip) tooltip.style.display = 'none';
            return;
        }

        // Create tooltip element if it doesn't exist
        if (!tooltip) {
            tooltip = document.createElement('div');
            tooltip.id = 'timeline-tooltip';
            tooltip.style.cssText = `
                position: fixed;
                background: #161b22;
                border: 1px solid #30363d;
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 12px;
                color: #f0f6fc;
                pointer-events: none;
                z-index: 1000;
                box-shadow: 0 4px 12px rgba(0,0,0,0.4);
            `;
            document.body.appendChild(tooltip);
        }

        const event = this.hoveredSpan.event;
        const dur = event.dur || 0;

        // Format duration nicely
        let durStr;
        if (dur >= 1) {
            durStr = dur.toFixed(2) + 's';
        } else if (dur >= 0.001) {
            durStr = (dur * 1000).toFixed(1) + 'ms';
        } else {
            durStr = (dur * 1000000).toFixed(0) + 'µs';
        }

        tooltip.innerHTML = `
            <div style="font-weight: 600; margin-bottom: 4px;">${event.name}</div>
            <div style="color: #8b949e;">Duration: <span style="color: #f0f6fc;">${durStr}</span></div>
        `;

        tooltip.style.display = 'block';
        tooltip.style.left = (this.tooltipX + 12) + 'px';
        tooltip.style.top = (this.tooltipY + 12) + 'px';

        // Keep tooltip in viewport
        const rect = tooltip.getBoundingClientRect();
        if (rect.right > window.innerWidth) {
            tooltip.style.left = (this.tooltipX - rect.width - 12) + 'px';
        }
        if (rect.bottom > window.innerHeight) {
            tooltip.style.top = (this.tooltipY - rect.height - 12) + 'px';
        }
    }

    updateFatesTooltip() {
        let tooltip = document.getElementById('fates-tooltip');

        if (!this.hoveredFate) {
            if (tooltip) tooltip.style.display = 'none';
            return;
        }

        // Create tooltip element if it doesn't exist
        if (!tooltip) {
            tooltip = document.createElement('div');
            tooltip.id = 'fates-tooltip';
            tooltip.style.cssText = `
                position: fixed;
                background: #161b22;
                border: 1px solid #30363d;
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 12px;
                color: #c9d1d9;
                pointer-events: none;
                z-index: 1000;
                box-shadow: 0 4px 12px rgba(0,0,0,0.4);
            `;
            document.body.appendChild(tooltip);
        }

        const f = this.hoveredFate.fate;
        const usedPct = Math.round(100 * f.used / f.total);

        tooltip.innerHTML = `
            <div style="font-weight: 600; margin-bottom: 6px;">Version ${f.version}</div>
            <div style="display: grid; grid-template-columns: auto auto; gap: 2px 12px; font-size: 11px;">
                <span style="color: #3fb950;">Used:</span><span style="text-align: right;">${f.used}</span>
                <span style="color: #d29922;">Partial:</span><span style="text-align: right;">${f.partial}</span>
                <span style="color: #f85149;">Wasted:</span><span style="text-align: right;">${f.wasted}</span>
                <span style="color: #8b949e;">Filtered:</span><span style="text-align: right;">${f.filtered}</span>
            </div>
            <div style="margin-top: 6px; padding-top: 6px; border-top: 1px solid #30363d; color: #8b949e;">
                Total: ${f.total} (${usedPct}% used)
            </div>
        `;

        tooltip.style.display = 'block';
        tooltip.style.left = (this.tooltipX + 12) + 'px';
        tooltip.style.top = (this.tooltipY + 12) + 'px';

        // Keep tooltip in viewport
        const rect = tooltip.getBoundingClientRect();
        if (rect.right > window.innerWidth) {
            tooltip.style.left = (this.tooltipX - rect.width - 12) + 'px';
        }
        if (rect.bottom > window.innerHeight) {
            tooltip.style.top = (this.tooltipY - rect.height - 12) + 'px';
        }
    }

    renderRewardVsLen() {
        const canvas = document.getElementById('reward-vs-len-canvas');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const rect = canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        const width = rect.width;
        const height = rect.height;

        if (width <= 0 || height <= 0) return;

        canvas.width = Math.floor(width * dpr);
        canvas.height = Math.floor(height * dpr);
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        // Clear
        ctx.fillStyle = '#0d1117';
        ctx.fillRect(0, 0, width, height);

        const data = this.rewardVsLen;
        if (data.length === 0) {
            ctx.fillStyle = '#8b949e';
            ctx.font = '14px -apple-system, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Reward vs Completion Length scatter will appear here', width / 2, height / 2);
            return;
        }

        const padding = { left: 50, right: 20, top: 20, bottom: 40 };
        const plotWidth = width - padding.left - padding.right;
        const plotHeight = height - padding.top - padding.bottom;

        // Find data ranges
        const lens = data.map(d => d.len);
        const rewards = data.map(d => d.reward);
        const minLen = Math.min(...lens);
        const maxLen = Math.max(...lens);
        const minReward = Math.min(...rewards);
        const maxReward = Math.max(...rewards);

        // Add padding to ranges
        const lenRange = maxLen - minLen || 1;
        const rewardRange = maxReward - minReward || 1;

        // Coordinate transforms
        const lenToX = (len) => padding.left + ((len - minLen) / lenRange) * plotWidth;
        const rewardToY = (reward) => height - padding.bottom - ((reward - minReward) / rewardRange) * plotHeight;

        // Draw axes
        ctx.strokeStyle = '#30363d';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(padding.left, padding.top);
        ctx.lineTo(padding.left, height - padding.bottom);
        ctx.lineTo(width - padding.right, height - padding.bottom);
        ctx.stroke();

        // Y-axis labels
        ctx.fillStyle = '#6e7681';
        ctx.font = '10px SF Mono, Monaco, monospace';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        ctx.fillText(maxReward.toFixed(1), padding.left - 5, padding.top);
        ctx.fillText(minReward.toFixed(1), padding.left - 5, height - padding.bottom);

        // X-axis labels
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        ctx.fillText(minLen.toString(), padding.left, height - padding.bottom + 5);
        ctx.fillText(maxLen.toString(), width - padding.right, height - padding.bottom + 5);
        ctx.fillText('Completion Length', width / 2, height - 15);

        // Y-axis label
        ctx.save();
        ctx.translate(12, height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.textAlign = 'center';
        ctx.fillText('Reward', 0, 0);
        ctx.restore();

        // Draw points - use transparency for overlap visualization
        // Color by reward: red for 0, green for 1
        const maxPoints = 5000;  // Limit points for performance
        const step = Math.max(1, Math.floor(data.length / maxPoints));

        for (let i = 0; i < data.length; i += step) {
            const d = data[i];
            const x = lenToX(d.len);
            const y = rewardToY(d.reward);

            // Color gradient: red (0) -> yellow (0.5) -> green (1)
            let r, g, b;
            if (d.reward <= 0.5) {
                const t = d.reward * 2;  // 0-1 for first half
                r = 248;  // red to yellow
                g = Math.round(81 + (210 - 81) * t);
                b = Math.round(73 + (34 - 73) * t);
            } else {
                const t = (d.reward - 0.5) * 2;  // 0-1 for second half
                r = Math.round(210 + (63 - 210) * t);  // yellow to green
                g = Math.round(153 + (185 - 153) * t);
                b = Math.round(34 + (80 - 34) * t);
            }

            ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.5)`;
            ctx.beginPath();
            ctx.arc(x, y, 2.5, 0, Math.PI * 2);
            ctx.fill();
        }

        // Show point count
        ctx.fillStyle = '#8b949e';
        ctx.font = '10px SF Mono, Monaco, monospace';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'top';
        ctx.fillText(`n=${data.length.toLocaleString()}`, width - padding.right, padding.top);
    }

    async showRolloutModal(promptId, version) {
        // Fetch rollout data from server
        const url = `/rollout?prompt_id=${encodeURIComponent(promptId)}${version !== undefined && version !== null ? `&version=${version}` : ''}`;

        try {
            const response = await fetch(url);
            if (!response.ok) {
                console.error(`Failed to fetch rollout: ${response.status}`);
                return;
            }
            const rollout = await response.json();
            this.renderRolloutModal(rollout);
        } catch (err) {
            console.error('Error fetching rollout:', err);
        }
    }

    renderRolloutModal(rollout) {
        // Remove existing modal if any
        const existingModal = document.getElementById('rollout-modal');
        if (existingModal) existingModal.remove();

        // Create modal backdrop
        const backdrop = document.createElement('div');
        backdrop.id = 'rollout-modal';
        backdrop.className = 'modal-backdrop';

        // Create modal content
        const modal = document.createElement('div');
        modal.className = 'modal-content';

        // Header
        const header = document.createElement('div');
        header.className = 'modal-header';
        header.innerHTML = `
            <div>
                <div class="modal-title">Rollout Inspection</div>
                <div class="modal-subtitle">${rollout.prompt_id} · v${rollout.version}</div>
            </div>
            <button class="modal-close">&times;</button>
        `;
        modal.appendChild(header);

        // Prompt section
        const promptSection = document.createElement('div');
        promptSection.className = 'modal-section';
        promptSection.innerHTML = `
            <div class="modal-section-title">Prompt</div>
            <pre class="modal-prompt">${this.escapeHtml(rollout.prompt)}</pre>
        `;
        modal.appendChild(promptSection);

        // Completions section
        const completionsSection = document.createElement('div');
        completionsSection.className = 'modal-section';

        const numPassed = rollout.rewards.filter(r => r > 0).length;
        const numTotal = rollout.rewards.length;

        completionsSection.innerHTML = `
            <div class="modal-section-title">Completions (${numPassed}/${numTotal} passed)</div>
        `;

        const completionsList = document.createElement('div');
        completionsList.className = 'modal-completions';

        for (let i = 0; i < rollout.completions.length; i++) {
            const completion = rollout.completions[i];
            const reward = rollout.rewards[i];
            const passed = reward > 0;

            const item = document.createElement('div');
            item.className = `modal-completion ${passed ? 'passed' : 'failed'}`;

            item.innerHTML = `
                <div class="completion-header">
                    <span class="completion-status">${passed ? '✓' : '✗'}</span>
                    <span class="completion-reward">${reward.toFixed(2)}</span>
                </div>
                <pre class="completion-text">${this.escapeHtml(this.truncateText(completion, 1000))}</pre>
            `;

            completionsList.appendChild(item);
        }

        completionsSection.appendChild(completionsList);
        modal.appendChild(completionsSection);

        backdrop.appendChild(modal);
        document.body.appendChild(backdrop);

        // Close on backdrop click or close button
        backdrop.addEventListener('click', (e) => {
            if (e.target === backdrop) backdrop.remove();
        });
        modal.querySelector('.modal-close').addEventListener('click', () => backdrop.remove());

        // Close on Escape
        const escHandler = (e) => {
            if (e.key === 'Escape') {
                backdrop.remove();
                document.removeEventListener('keydown', escHandler);
            }
        };
        document.addEventListener('keydown', escHandler);
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    truncateText(text, maxLen) {
        if (text.length <= maxLen) return text;
        return text.slice(0, maxLen) + '\n... [truncated]';
    }
}

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    window.heartbeat = new HeartbeatViz();
});
