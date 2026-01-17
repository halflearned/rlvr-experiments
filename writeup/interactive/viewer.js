/**
 * RLVR Training Trace Viewer
 * Renders pipeline timeline, buffer dynamics, and sample fates from trace data.
 */

class TraceViewer {
    constructor() {
        // Canvas contexts
        this.timelineCtx = null;
        this.bufferCtx = null;
        this.fatesCtx = null;

        // Trace data
        this.events = [];
        this.maxTs = 0;
        this.playhead = 0;  // Current time position

        // Parsed event arrays
        this.vllmSpans = [];
        this.refSpans = [];
        this.trainerSpans = [];
        this.verifierSpans = [];
        this.syncEvents = [];
        this.bufferEvents = [];
        this.optimEvents = [];

        // Playback
        this.playing = false;
        this.playInterval = null;

        // Colors
        this.colors = {
            generate: '#3fb950',
            reference: '#58a6ff',
            trainer: '#a371f7',
            verifier: '#f0883e',
            sync: '#f85149',
            buffer: ['#3fb950', '#58a6ff', '#a371f7', '#f0883e', '#39d4e0',
                     '#d29922', '#f85149', '#8b949e', '#2ea88a', '#bc8cff']
        };

        this.init();
    }

    async init() {
        // Get canvas contexts
        const timelineCanvas = document.getElementById('timeline-canvas');
        const bufferCanvas = document.getElementById('buffer-canvas');
        const fatesCanvas = document.getElementById('fates-canvas');

        this.timelineCtx = timelineCanvas.getContext('2d');
        this.bufferCtx = bufferCanvas.getContext('2d');
        this.fatesCtx = fatesCanvas.getContext('2d');

        // Setup controls
        this.setupControls();

        // Load trace data
        await this.loadTrace('trace.jsonl');

        // Initial render
        this.render();
    }

    setupControls() {
        const slider = document.getElementById('time-slider');
        const prevBtn = document.getElementById('prev-btn');
        const nextBtn = document.getElementById('next-btn');
        const playBtn = document.getElementById('play-btn');

        slider.addEventListener('input', (e) => {
            this.playhead = (e.target.value / 100) * this.maxTs;
            this.updateTimeDisplay();
            this.render();
        });

        prevBtn.addEventListener('click', () => {
            this.step(-1);
        });

        nextBtn.addEventListener('click', () => {
            this.step(1);
        });

        playBtn.addEventListener('click', () => {
            this.togglePlay();
        });

        // Handle window resize
        window.addEventListener('resize', () => this.render());
    }

    step(direction) {
        // Step by ~5 seconds
        const stepSize = 5;
        this.playhead = Math.max(0, Math.min(this.maxTs, this.playhead + direction * stepSize));
        this.updateSlider();
        this.updateTimeDisplay();
        this.render();
    }

    togglePlay() {
        this.playing = !this.playing;
        const playBtn = document.getElementById('play-btn');
        playBtn.innerHTML = this.playing ? '&#10074;&#10074;' : '&#9654;';

        if (this.playing) {
            this.playInterval = setInterval(() => {
                this.playhead += 0.5;  // Advance 0.5s per frame
                if (this.playhead >= this.maxTs) {
                    this.playhead = this.maxTs;
                    this.togglePlay();
                }
                this.updateSlider();
                this.updateTimeDisplay();
                this.render();
            }, 50);  // 20fps
        } else {
            clearInterval(this.playInterval);
        }
    }

    updateSlider() {
        const slider = document.getElementById('time-slider');
        slider.value = (this.playhead / this.maxTs) * 100;
    }

    updateTimeDisplay() {
        const display = document.getElementById('time-display');
        display.textContent = `${this.playhead.toFixed(1)}s / ${this.maxTs.toFixed(0)}s`;
    }

    async loadTrace(url) {
        try {
            const response = await fetch(url);
            const text = await response.text();
            const lines = text.trim().split('\n');

            // Parse JSON lines
            for (const line of lines) {
                if (!line.trim()) continue;
                try {
                    const event = JSON.parse(line);
                    this.events.push(event);
                } catch (e) {
                    // Skip malformed lines
                }
            }

            this.processEvents();
            this.updateTimeDisplay();
        } catch (e) {
            console.error('Failed to load trace:', e);
        }
    }

    processEvents() {
        let currentVersion = 0;

        for (const event of this.events) {
            const ts = event.ts || 0;
            this.maxTs = Math.max(this.maxTs, ts + (event.dur || 0));

            // Track version from sync events
            if (event.type === 'sync') {
                currentVersion = event.version || currentVersion + 1;
                this.syncEvents.push({ ts, version: currentVersion });
            }

            // Span events (pipeline activity)
            if (event.type === 'span') {
                const span = {
                    ts,
                    dur: event.dur || 0,
                    name: event.name,
                    replica: event.replica || 0,
                    slot: event.slot || 0,
                    worker: event.worker || 0
                };

                if (event.name === 'vllm.generate_single' || event.name === 'generate') {
                    this.vllmSpans.push(span);
                } else if (event.name === 'reference' || event.name === 'ref_logprobs' || event.name === 'vllm.get_logprobs') {
                    this.refSpans.push(span);
                } else if (event.name === 'train' || event.name === 'forward_backward' || event.name === 'optim_step') {
                    this.trainerSpans.push(span);
                } else if (event.name === 'verify' || event.name === 'verifier') {
                    this.verifierSpans.push(span);
                } else if (event.name.startsWith('sync.')) {
                    // Treat sync spans as sync events
                    this.syncEvents.push({ ts, version: currentVersion, dur: event.dur });
                }
            }

            // Buffer events
            if (event.type === 'buffer') {
                this.bufferEvents.push({
                    ts,
                    size: event.size,
                    byVersion: event.by_version || {},
                    fates: event.fates || { used: {}, wasted: {}, filtered: {}, failed: {} },
                    version: currentVersion
                });
            }

            // Optim events
            if (event.type === 'optim') {
                this.optimEvents.push({ ts, step: event.step });
            }
        }

        // Set playhead to first generation event
        if (this.vllmSpans.length > 0) {
            this.playhead = this.vllmSpans[0].ts + 5;  // Start 5s after first generation
        } else {
            this.playhead = Math.min(30, this.maxTs * 0.01);
        }
    }

    render() {
        this.renderTimeline();
        this.renderBuffer();
        this.renderFates();
    }

    renderTimeline() {
        const canvas = document.getElementById('timeline-canvas');
        const ctx = this.timelineCtx;
        const rect = canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;

        canvas.width = Math.floor(rect.width * dpr);
        canvas.height = Math.floor(rect.height * dpr);
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        const width = rect.width;
        const height = rect.height;

        // Clear
        ctx.fillStyle = '#0d1117';
        ctx.fillRect(0, 0, width, height);

        const padding = { left: 80, right: 20, top: 10, bottom: 25 };
        const plotWidth = width - padding.left - padding.right;
        const plotHeight = height - padding.top - padding.bottom;

        // Fixed x-axis from 180s to maxTs
        const visibleStart = 180;
        const visibleEnd = this.maxTs;
        const duration = visibleEnd - visibleStart;

        const timeToX = (t) => padding.left + ((t - visibleStart) / duration) * plotWidth;

        // Lane layout
        const lanes = ['Generation', 'Verifier', 'Reference', 'Training'];
        const laneHeight = plotHeight / lanes.length;

        // Draw lane labels and backgrounds
        ctx.font = '11px -apple-system, sans-serif';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';

        for (let i = 0; i < lanes.length; i++) {
            const y = padding.top + i * laneHeight;
            ctx.fillStyle = i % 2 === 0 ? '#161b22' : '#0d1117';
            ctx.fillRect(padding.left, y, plotWidth, laneHeight);
            ctx.fillStyle = '#8b949e';
            ctx.fillText(lanes[i], padding.left - 8, y + laneHeight / 2);
        }

        // Draw spans
        const drawSpans = (spans, laneIndex, color) => {
            const y = padding.top + laneIndex * laneHeight + 4;
            const h = laneHeight - 8;
            ctx.fillStyle = color;

            for (const span of spans) {
                if (span.ts > visibleEnd || span.ts + span.dur < visibleStart) continue;
                if (span.ts > this.playhead) continue;  // Only show up to playhead

                const x1 = Math.max(padding.left, timeToX(span.ts));
                const x2 = Math.min(padding.left + plotWidth, timeToX(Math.min(span.ts + span.dur, this.playhead)));
                if (x2 > x1) {
                    ctx.fillRect(x1, y, x2 - x1, h);
                }
            }
        };

        drawSpans(this.vllmSpans, 0, this.colors.generate);
        drawSpans(this.verifierSpans, 1, this.colors.verifier);
        drawSpans(this.refSpans, 2, this.colors.reference);
        drawSpans(this.trainerSpans, 3, this.colors.trainer);
        // Order: Generation -> Verifier -> Reference -> Training (matches pipeline flow)

        // Draw sync events as vertical lines
        ctx.strokeStyle = this.colors.sync;
        ctx.lineWidth = 2;
        for (const sync of this.syncEvents) {
            if (sync.ts < visibleStart || sync.ts > visibleEnd) continue;
            if (sync.ts > this.playhead) continue;
            const x = timeToX(sync.ts);
            ctx.beginPath();
            ctx.moveTo(x, padding.top);
            ctx.lineTo(x, padding.top + plotHeight);
            ctx.stroke();
        }

        // Draw playhead
        const playheadX = timeToX(this.playhead);
        if (playheadX >= padding.left && playheadX <= padding.left + plotWidth) {
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 2;
            ctx.setLineDash([4, 4]);
            ctx.beginPath();
            ctx.moveTo(playheadX, padding.top);
            ctx.lineTo(playheadX, padding.top + plotHeight);
            ctx.stroke();
            ctx.setLineDash([]);
        }

        // Time axis
        ctx.fillStyle = '#8b949e';
        ctx.font = '10px -apple-system, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        const numTicks = 6;
        for (let i = 0; i <= numTicks; i++) {
            const t = visibleStart + (duration * i / numTicks);
            const x = timeToX(t);
            ctx.fillText(t.toFixed(0) + 's', x, height - 15);
        }
    }

    renderBuffer() {
        const canvas = document.getElementById('buffer-canvas');
        const ctx = this.bufferCtx;
        const rect = canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;

        canvas.width = Math.floor(rect.width * dpr);
        canvas.height = Math.floor(rect.height * dpr);
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        const width = rect.width;
        const height = rect.height;

        // Clear
        ctx.fillStyle = '#0d1117';
        ctx.fillRect(0, 0, width, height);

        if (!this.bufferEvents.length) {
            ctx.fillStyle = '#8b949e';
            ctx.font = '14px -apple-system, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('No buffer data', width / 2, height / 2);
            return;
        }

        const padding = { left: 50, right: 20, top: 10, bottom: 25 };
        const plotWidth = width - padding.left - padding.right;
        const plotHeight = height - padding.top - padding.bottom;

        // Fixed x-axis from 180s to maxTs (matching timeline)
        const visibleStart = 180;
        const visibleEnd = this.maxTs;
        const duration = visibleEnd - visibleStart;

        const timeToX = (t) => padding.left + ((t - visibleStart) / duration) * plotWidth;

        // Filter events up to playhead
        const visibleEvents = this.bufferEvents.filter(e => e.ts <= this.playhead && e.ts >= visibleStart - 10);

        if (!visibleEvents.length) {
            ctx.fillStyle = '#8b949e';
            ctx.font = '14px -apple-system, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Buffer data will appear here', width / 2, height / 2);
            return;
        }

        // Collect versions
        const allVersions = new Set();
        for (const evt of visibleEvents) {
            for (const v of Object.keys(evt.byVersion || {})) {
                allVersions.add(parseInt(v));
            }
        }
        const versions = [...allVersions].sort((a, b) => a - b);

        // Max size for scaling
        let maxSize = Math.max(...visibleEvents.map(e => e.size), 1);
        maxSize = Math.max(maxSize, 50);  // Minimum scale

        // Draw stacked area
        if (versions.length > 0) {
            const stackedData = visibleEvents.map(evt => {
                const result = [];
                let cumulative = 0;
                for (const version of versions) {
                    const count = evt.byVersion[version] || 0;
                    result.push({ bottom: cumulative, top: cumulative + count, count });
                    cumulative += count;
                }
                return result;
            });

            // Draw each version band
            for (let vi = 0; vi < versions.length; vi++) {
                const version = versions[vi];
                const color = this.colors.buffer[version % this.colors.buffer.length];

                const points = [];
                for (let i = 0; i < visibleEvents.length; i++) {
                    const evt = visibleEvents[i];
                    const stack = stackedData[i][vi];
                    points.push({
                        x: timeToX(evt.ts),
                        bottom: height - padding.bottom - (stack.bottom / maxSize) * plotHeight,
                        top: height - padding.bottom - (stack.top / maxSize) * plotHeight
                    });
                }

                if (points.length < 2) continue;

                // Draw filled band
                ctx.beginPath();
                ctx.moveTo(points[0].x, points[0].bottom);
                for (let i = 0; i < points.length; i++) {
                    if (i > 0) ctx.lineTo(points[i].x, points[i - 1].top);
                    ctx.lineTo(points[i].x, points[i].top);
                }
                for (let i = points.length - 1; i >= 0; i--) {
                    if (i < points.length - 1) ctx.lineTo(points[i].x, points[i + 1].bottom);
                    ctx.lineTo(points[i].x, points[i].bottom);
                }
                ctx.closePath();
                ctx.fillStyle = color + 'cc';
                ctx.fill();
            }
        }

        // Draw sync events
        ctx.strokeStyle = this.colors.sync + '66';
        ctx.lineWidth = 2;
        for (const sync of this.syncEvents) {
            if (sync.ts < visibleStart || sync.ts > visibleEnd) continue;
            if (sync.ts > this.playhead) continue;
            const x = timeToX(sync.ts);
            ctx.beginPath();
            ctx.moveTo(x, padding.top);
            ctx.lineTo(x, height - padding.bottom);
            ctx.stroke();
        }

        // Y-axis label
        ctx.fillStyle = '#8b949e';
        ctx.font = '10px -apple-system, sans-serif';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        ctx.fillText(maxSize.toFixed(0), padding.left - 8, padding.top);
        ctx.fillText('0', padding.left - 8, height - padding.bottom);

        // Time axis
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        const numTicks = 6;
        for (let i = 0; i <= numTicks; i++) {
            const t = visibleStart + (duration * i / numTicks);
            const x = timeToX(t);
            ctx.fillText(t.toFixed(0) + 's', x, height - 15);
        }
    }

    renderFates() {
        const canvas = document.getElementById('fates-canvas');
        const ctx = this.fatesCtx;
        const rect = canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;

        canvas.width = Math.floor(rect.width * dpr);
        canvas.height = Math.floor(rect.height * dpr);
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        const width = rect.width;
        const height = rect.height;

        // Clear
        ctx.fillStyle = '#0d1117';
        ctx.fillRect(0, 0, width, height);

        // Get latest fates up to playhead
        const relevantEvents = this.bufferEvents.filter(e => e.ts <= this.playhead);
        if (!relevantEvents.length) {
            ctx.fillStyle = '#8b949e';
            ctx.font = '14px -apple-system, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Sample fates will appear here', width / 2, height / 2);
            return;
        }

        const lastEvent = relevantEvents[relevantEvents.length - 1];
        const fates = lastEvent.fates || { used: {}, wasted: {}, filtered: {}, failed: {} };

        // Sum up fates
        const sumFates = (obj) => Object.values(obj).reduce((a, b) => a + b, 0);
        const used = sumFates(fates.used);
        const wasted = sumFates(fates.wasted);
        const filtered = sumFates(fates.filtered);
        const failed = sumFates(fates.failed);
        const total = used + wasted + filtered + failed;

        if (total === 0) {
            ctx.fillStyle = '#8b949e';
            ctx.font = '14px -apple-system, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('No samples processed yet', width / 2, height / 2);
            return;
        }

        const padding = { left: 60, right: 20, top: 20, bottom: 40 };
        const barWidth = width - padding.left - padding.right;
        const barHeight = 40;
        const barY = height / 2 - barHeight / 2;

        // Draw stacked horizontal bar
        const categories = [
            { label: 'Used', value: used, color: '#3fb950' },
            { label: 'Filtered', value: filtered, color: '#8b949e' },
            { label: 'Wasted', value: wasted, color: '#f0883e' },
            { label: 'Failed', value: failed, color: '#f85149' }
        ];

        let x = padding.left;
        for (const cat of categories) {
            if (cat.value === 0) continue;
            const w = (cat.value / total) * barWidth;
            ctx.fillStyle = cat.color;
            ctx.fillRect(x, barY, w, barHeight);
            x += w;
        }

        // Labels below
        ctx.font = '12px -apple-system, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';

        x = padding.left;
        for (const cat of categories) {
            if (cat.value === 0) continue;
            const w = (cat.value / total) * barWidth;
            const pct = ((cat.value / total) * 100).toFixed(0);
            ctx.fillStyle = cat.color;
            ctx.fillText(`${cat.label}: ${cat.value} (${pct}%)`, x + w / 2, barY + barHeight + 8);
            x += w;
        }

        // Total label
        ctx.fillStyle = '#8b949e';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        ctx.fillText(`Total: ${total}`, padding.left - 10, height / 2);
    }
}

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    new TraceViewer();
});
