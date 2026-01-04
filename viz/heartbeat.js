// RLVR Heartbeat Visualization

class HeartbeatViz {
    constructor() {
        this.data = null;
        this.events = [];
        this.minTs = 0;
        this.maxTs = 0;

        // Canvas contexts
        this.timelineCtx = null;
        this.bufferCtx = null;
        this.metricCharts = {};

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
            diff_rollout: []
        };

        // Metrics that use log scale
        this.logScaleMetrics = new Set(['grad_norm']);

        // Hover state for tooltips
        this.hoverMetric = null;
        this.hoverX = 0;

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

        // Try to load default trace
        this.loadDefaultTrace();
    }

    setupCanvases() {
        const timelineCanvas = document.getElementById('timeline-canvas');
        const bufferCanvas = document.getElementById('buffer-canvas');

        this.timelineCtx = timelineCanvas.getContext('2d');
        this.bufferCtx = bufferCanvas.getContext('2d');

        // Setup metric mini-charts with hover handling
        const metricNames = Object.keys(this.metrics);
        metricNames.forEach(metric => {
            const canvas = document.getElementById(`chart-${metric}`);
            if (canvas) {
                this.metricCharts[metric] = canvas.getContext('2d');

                // Add hover listeners
                canvas.addEventListener('mousemove', (e) => {
                    const rect = canvas.getBoundingClientRect();
                    this.hoverMetric = metric;
                    this.hoverX = (e.clientX - rect.left) / rect.width;
                    this.renderMetricChart(metric);
                });
                canvas.addEventListener('mouseleave', () => {
                    this.hoverMetric = null;
                    this.renderMetricChart(metric);
                });
            }
        });
    }

    async loadDefaultTrace() {
        try {
            // Try loading from common locations (now JSONL format)
            const paths = [
                '../trace.jsonl',
                '/efs/rlvr-experiments/trace.jsonl',
                '../traces/trace.jsonl'
            ];

            for (const path of paths) {
                try {
                    const response = await fetch(path);
                    if (response.ok) {
                        const text = await response.text();
                        this.processJSONL(text);
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
            kl_max: [], ratio_max: [], diff_ref: [], diff_rollout: []
        };
        this.bufferEvents = [];
        this.syncEvents = [];

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
            }

            // Span events for timeline
            if (event.type === 'span') {
                // Track sync events for version changes
                if (event.name === 'sync.titan_to_vllm') {
                    currentVersion++;
                    this.syncEvents.push({ ts, version: currentVersion });
                }
            }

            // Buffer events (new specialized type)
            if (event.type === 'buffer') {
                this.bufferEvents.push({
                    ts,
                    size: event.size,
                    byVersion: event.by_version || {},
                    evicted: event.evicted || {},
                    version: currentVersion
                });
            }
        }
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
        this.renderTimeline();
        this.renderBuffer();
        this.renderMetrics();
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

        // Define swimlanes (new format uses type='span')
        const lanes = [
            { name: 'vLLM 0', filter: e => e.type === 'span' && e.name === 'vllm.generate' && e.replica === 0 },
            { name: 'vLLM 1', filter: e => e.type === 'span' && e.name === 'vllm.generate' && e.replica === 1 },
            { name: 'vLLM 2', filter: e => e.type === 'span' && e.name === 'vllm.generate' && e.replica === 2 },
            { name: 'vLLM 3', filter: e => e.type === 'span' && e.name === 'vllm.generate' && e.replica === 3 },
            { name: 'Trainer', filter: e => e.type === 'span' && e.name === 'forward_backward' },
            { name: 'Sync', filter: e => e.type === 'span' && e.name.startsWith('sync.') }
        ];

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
            'forward_backward': '#58a6ff',
            'sync.trainer_to_vllm': '#f85149',
            'sync.trainer_to_reference': '#f0883e',
            'sync.waiting_for_vllm_pause': '#f8514966',
            'sync.titan_to_vllm': '#f85149',
            'sync.titan_to_titan': '#f0883e'
        };

        for (const event of this.events) {
            if (event.type !== 'span') continue;

            const laneIdx = lanes.findIndex(l => l.filter(event));
            if (laneIdx === -1) continue;

            const startTs = event.ts;  // Already in seconds
            const dur = event.dur || 0;

            const x = padding.left + (startTs / duration) * plotWidth;
            const w = Math.max(1, (dur / duration) * plotWidth);
            const y = padding.top + laneIdx * laneHeight + 4;
            const h = laneHeight - 8;

            ctx.fillStyle = colors[event.name] || '#8b949e';
            ctx.fillRect(x, y, w, h);
        }

        // Draw time axis
        ctx.fillStyle = '#8b949e';
        ctx.font = '10px -apple-system, sans-serif';
        ctx.textAlign = 'center';

        const numTicks = 10;
        for (let i = 0; i <= numTicks; i++) {
            const t = (duration * i / numTicks);
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

        // Draw stacked area chart
        if (versions.length > 0) {
            // For each version, draw its contribution as a filled area
            // We need to draw from bottom up (oldest version first)
            for (let vi = 0; vi < versions.length; vi++) {
                const version = versions[vi];
                const color = versionColors[version % versionColors.length];

                ctx.beginPath();
                ctx.moveTo(padding.left, height - padding.bottom);

                // First pass: draw top edge
                for (let i = 0; i < this.bufferEvents.length; i++) {
                    const evt = this.bufferEvents[i];
                    const x = padding.left + (evt.ts / duration) * plotWidth;

                    // Sum of this version and all older versions
                    let stackedHeight = 0;
                    for (let vj = 0; vj <= vi; vj++) {
                        stackedHeight += (evt.byVersion[versions[vj]] || 0);
                    }
                    const y = height - padding.bottom - (stackedHeight / maxSize) * plotHeight;
                    ctx.lineTo(x, y);
                }

                // Second pass: draw bottom edge (going backwards)
                for (let i = this.bufferEvents.length - 1; i >= 0; i--) {
                    const evt = this.bufferEvents[i];
                    const x = padding.left + (evt.ts / duration) * plotWidth;

                    // Sum of all versions below this one
                    let stackedHeight = 0;
                    for (let vj = 0; vj < vi; vj++) {
                        stackedHeight += (evt.byVersion[versions[vj]] || 0);
                    }
                    const y = height - padding.bottom - (stackedHeight / maxSize) * plotHeight;
                    ctx.lineTo(x, y);
                }

                ctx.closePath();
                ctx.fillStyle = color + '99';  // Semi-transparent
                ctx.fill();
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

        // Draw total size outline
        ctx.beginPath();
        for (let i = 0; i < this.bufferEvents.length; i++) {
            const evt = this.bufferEvents[i];
            const x = padding.left + (evt.ts / duration) * plotWidth;
            const y = height - padding.bottom - (evt.size / maxSize) * plotHeight;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = '#f0f6fc';
        ctx.lineWidth = 1;
        ctx.stroke();

        // Draw sync event vertical lines
        for (const sync of this.syncEvents) {
            const x = padding.left + (sync.ts / duration) * plotWidth;
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
            const t = (duration * i / numTicks);
            const x = padding.left + (i / numTicks) * plotWidth;
            ctx.fillText(t.toFixed(1) + 's', x, height - 10);
        }
    }

    renderMetrics() {
        for (const metric of Object.keys(this.metrics)) {
            this.renderMetricChart(metric);
        }
    }

    renderMetricChart(metric) {
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
            diff_rollout: '#39d4e0'
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
        const lastValue = data[data.length - 1].value;
        const el = document.getElementById(`metric-${metric}`);
        if (el) {
            el.textContent = this.formatMetricValue(metric, lastValue);
        }

        // Determine if log scale
        const useLog = this.logScaleMetrics.has(metric);

        // Get values, apply log if needed
        const rawValues = data.map(d => d.value);
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

        // Draw sparkline
        ctx.beginPath();
        for (let i = 0; i < data.length; i++) {
            const x = padding.left + (i / (data.length - 1 || 1)) * plotWidth;
            const y = valueToY(data[i].value);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }

        ctx.strokeStyle = colors[metric];
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Draw hover indicator and value
        if (this.hoverMetric === metric && data.length > 0) {
            const idx = Math.min(Math.floor(this.hoverX * data.length), data.length - 1);
            const x = padding.left + (idx / (data.length - 1 || 1)) * plotWidth;
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
                return value.toFixed(1) + '%';
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
}

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    window.heartbeat = new HeartbeatViz();
});
