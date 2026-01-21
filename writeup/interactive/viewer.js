/**
 * RLVR Training Trace Viewer
 * Renders pipeline timeline, buffer dynamics, and sample fates from trace data.
 *
 * Performance optimizations for large traces:
 * - Streaming parse with sampling for traces > 50k lines
 * - Lazy initialization - only loads data when visualization is opened
 * - Windowed viewing with navigation for timeline/buffer graphs
 */

class TraceViewer {
    constructor() {
        this.timelineCtx = null;
        this.bufferCtx = null;
        this.fatesCtx = null;

        this.dataLoaded = false;
        this.maxTs = 0;

        this.vllmSpans = [];
        this.refSpans = [];
        this.trainerSpans = [];
        this.verifierSpans = [];
        this.syncEvents = [];
        this.bufferEvents = [];

        // Windowed viewing
        this.windowDuration = 300;  // Show 300 seconds at a time
        this.timelineStart = 0;
        this.bufferStart = 0;

        // Sampling config
        this.maxSpansPerType = 5000;
        this.maxBufferEvents = 2000;

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
        const timelineCanvas = document.getElementById('timeline-canvas');
        const bufferCanvas = document.getElementById('buffer-canvas');
        const fatesCanvas = document.getElementById('fates-canvas');

        if (!timelineCanvas) return;

        this.timelineCtx = timelineCanvas.getContext('2d');
        this.bufferCtx = bufferCanvas.getContext('2d');
        this.fatesCtx = fatesCanvas.getContext('2d');

        this.setupNavigation();

        const details = document.querySelector('.visualization-details');
        if (details) {
            details.addEventListener('toggle', async () => {
                if (details.open) {
                    await this.loadTraceIfNeeded();
                }
            });
            if (details.open) {
                await this.loadTraceIfNeeded();
            }
        }

        let resizeTimeout;
        window.addEventListener('resize', () => {
            clearTimeout(resizeTimeout);
            resizeTimeout = setTimeout(() => {
                if (this.dataLoaded) this.render();
            }, 100);
        });
    }

    setupNavigation() {
        const timelinePrev = document.getElementById('timeline-prev');
        const timelineNext = document.getElementById('timeline-next');
        const bufferPrev = document.getElementById('buffer-prev');
        const bufferNext = document.getElementById('buffer-next');

        if (timelinePrev) {
            timelinePrev.addEventListener('click', () => {
                this.timelineStart = Math.max(0, this.timelineStart - this.windowDuration);
                this.renderTimeline();
            });
        }
        if (timelineNext) {
            timelineNext.addEventListener('click', () => {
                this.timelineStart = Math.min(
                    Math.max(0, this.maxTs - this.windowDuration),
                    this.timelineStart + this.windowDuration
                );
                this.renderTimeline();
            });
        }
        if (bufferPrev) {
            bufferPrev.addEventListener('click', () => {
                this.bufferStart = Math.max(0, this.bufferStart - this.windowDuration);
                this.renderBuffer();
            });
        }
        if (bufferNext) {
            bufferNext.addEventListener('click', () => {
                this.bufferStart = Math.min(
                    Math.max(0, this.maxTs - this.windowDuration),
                    this.bufferStart + this.windowDuration
                );
                this.renderBuffer();
            });
        }
    }

    async loadTraceIfNeeded() {
        if (this.dataLoaded) return;
        this.dataLoaded = true;
        await this.loadTrace('trace.jsonl');
        // Start at end of trace to show most recent data
        this.timelineStart = Math.max(0, this.maxTs - this.windowDuration);
        this.bufferStart = Math.max(0, this.maxTs - this.windowDuration);
        this.render();
    }

    async loadTrace(url) {
        try {
            const response = await fetch(url);
            const text = await response.text();
            const lines = text.trim().split('\n');

            console.log(`Trace has ${lines.length} lines`);

            const isLarge = lines.length > 50000;
            const sampleRate = isLarge ? Math.ceil(lines.length / 50000) : 1;

            if (isLarge) {
                console.log(`Large trace detected, sampling every ${sampleRate} events`);
            }

            await this.parseEventsChunked(lines, sampleRate);
        } catch (e) {
            console.error('Failed to load trace:', e);
        }
    }

    async parseEventsChunked(lines, sampleRate) {
        const chunkSize = 10000;
        let currentVersion = 0;
        let lineIndex = 0;

        const tempVllm = [];
        const tempRef = [];
        const tempTrainer = [];
        const tempVerifier = [];
        const tempBuffer = [];

        const processChunk = () => {
            const endIndex = Math.min(lineIndex + chunkSize, lines.length);

            for (; lineIndex < endIndex; lineIndex++) {
                const line = lines[lineIndex];
                if (!line || !line.trim()) continue;

                let event;
                try {
                    event = JSON.parse(line);
                } catch (e) {
                    continue;
                }

                const ts = event.ts || 0;
                this.maxTs = Math.max(this.maxTs, ts + (event.dur || 0));

                if (event.type === 'sync') {
                    currentVersion = event.version || currentVersion + 1;
                    this.syncEvents.push({ ts, version: currentVersion });
                    continue;
                }

                if (event.type === 'buffer') {
                    tempBuffer.push({
                        ts,
                        size: event.size,
                        byVersion: event.by_version || {},
                        fates: event.fates || { used: {}, wasted: {}, filtered: {}, failed: {} },
                        version: currentVersion
                    });
                    continue;
                }

                if (event.type === 'span' && (sampleRate === 1 || lineIndex % sampleRate === 0)) {
                    const span = { ts, dur: event.dur || 0 };

                    if (event.name === 'vllm.generate_single' || event.name === 'generate') {
                        tempVllm.push(span);
                    } else if (event.name === 'reference' || event.name === 'ref_logprobs' || event.name === 'vllm.get_logprobs') {
                        tempRef.push(span);
                    } else if (event.name === 'train' || event.name === 'forward_backward' || event.name === 'optim_step') {
                        tempTrainer.push(span);
                    } else if (event.name === 'verify' || event.name === 'verifier') {
                        tempVerifier.push(span);
                    } else if (event.name.startsWith('sync.')) {
                        this.syncEvents.push({ ts, version: currentVersion, dur: event.dur });
                    }
                }
            }
        };

        while (lineIndex < lines.length) {
            processChunk();
            if (lineIndex < lines.length) {
                await new Promise(resolve => setTimeout(resolve, 0));
            }
        }

        this.vllmSpans = this.downsampleSpans(tempVllm, this.maxSpansPerType);
        this.refSpans = this.downsampleSpans(tempRef, this.maxSpansPerType);
        this.trainerSpans = this.downsampleSpans(tempTrainer, this.maxSpansPerType);
        this.verifierSpans = this.downsampleSpans(tempVerifier, this.maxSpansPerType);
        this.bufferEvents = this.downsampleSpans(tempBuffer, this.maxBufferEvents);

        console.log(`Loaded: ${this.vllmSpans.length} gen, ${this.refSpans.length} ref, ${this.trainerSpans.length} train, ${this.verifierSpans.length} verify, ${this.bufferEvents.length} buffer`);
    }

    downsampleSpans(spans, maxCount) {
        if (spans.length <= maxCount) return spans;
        const step = spans.length / maxCount;
        const result = [];
        for (let i = 0; i < maxCount; i++) {
            result.push(spans[Math.floor(i * step)]);
        }
        return result;
    }

    render() {
        this.renderTimeline();
        this.renderBuffer();
        this.renderFates();
    }

    updateRangeDisplay(id, start, end, max) {
        const el = document.getElementById(id);
        if (el) {
            el.textContent = `${Math.floor(start)}s - ${Math.floor(end)}s / ${Math.floor(max)}s`;
        }
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

        ctx.fillStyle = '#0d1117';
        ctx.fillRect(0, 0, width, height);

        const padding = { left: 80, right: 20, top: 10, bottom: 25 };
        const plotWidth = width - padding.left - padding.right;
        const plotHeight = height - padding.top - padding.bottom;

        const visibleStart = this.timelineStart;
        const visibleEnd = Math.min(this.timelineStart + this.windowDuration, this.maxTs);
        const duration = visibleEnd - visibleStart;

        this.updateRangeDisplay('timeline-range', visibleStart, visibleEnd, this.maxTs);

        if (duration <= 0) return;

        const timeToX = (t) => padding.left + ((t - visibleStart) / duration) * plotWidth;

        const lanes = ['Generation', 'Verifier', 'Reference', 'Training'];
        const laneHeight = plotHeight / lanes.length;

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

        const drawSpans = (spans, laneIndex, color) => {
            const y = padding.top + laneIndex * laneHeight + 4;
            const h = laneHeight - 8;
            ctx.fillStyle = color;

            for (const span of spans) {
                if (span.ts > visibleEnd || span.ts + span.dur < visibleStart) continue;
                const x1 = Math.max(padding.left, timeToX(span.ts));
                const x2 = Math.min(padding.left + plotWidth, timeToX(span.ts + span.dur));
                if (x2 > x1) {
                    ctx.fillRect(x1, y, Math.max(1, x2 - x1), h);
                }
            }
        };

        drawSpans(this.vllmSpans, 0, this.colors.generate);
        drawSpans(this.verifierSpans, 1, this.colors.verifier);
        drawSpans(this.refSpans, 2, this.colors.reference);
        drawSpans(this.trainerSpans, 3, this.colors.trainer);

        ctx.strokeStyle = this.colors.sync;
        ctx.lineWidth = 1;
        for (const sync of this.syncEvents) {
            if (sync.ts < visibleStart || sync.ts > visibleEnd) continue;
            const x = timeToX(sync.ts);
            ctx.beginPath();
            ctx.moveTo(x, padding.top);
            ctx.lineTo(x, padding.top + plotHeight);
            ctx.stroke();
        }

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

        ctx.fillStyle = '#0d1117';
        ctx.fillRect(0, 0, width, height);

        const visibleStart = this.bufferStart;
        const visibleEnd = Math.min(this.bufferStart + this.windowDuration, this.maxTs);

        this.updateRangeDisplay('buffer-range', visibleStart, visibleEnd, this.maxTs);

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

        const duration = visibleEnd - visibleStart;
        if (duration <= 0) return;

        const timeToX = (t) => padding.left + ((t - visibleStart) / duration) * plotWidth;

        // Filter to visible range
        const visibleEvents = this.bufferEvents.filter(
            e => e.ts >= visibleStart && e.ts <= visibleEnd
        );

        if (!visibleEvents.length) {
            ctx.fillStyle = '#8b949e';
            ctx.font = '14px -apple-system, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('No data in range', width / 2, height / 2);
            return;
        }

        const allVersions = new Set();
        for (const evt of visibleEvents) {
            for (const v of Object.keys(evt.byVersion || {})) {
                allVersions.add(parseInt(v));
            }
        }
        const versions = [...allVersions].sort((a, b) => a - b);

        let maxSize = Math.max(...visibleEvents.map(e => e.size), 1);
        maxSize = Math.max(maxSize, 50);

        if (versions.length > 0) {
            const stackedData = visibleEvents.map(evt => {
                const result = [];
                let cumulative = 0;
                for (const version of versions) {
                    const count = evt.byVersion[version] || 0;
                    result.push({ bottom: cumulative, top: cumulative + count });
                    cumulative += count;
                }
                return result;
            });

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

        ctx.strokeStyle = this.colors.sync + '66';
        ctx.lineWidth = 1;
        for (const sync of this.syncEvents) {
            if (sync.ts < visibleStart || sync.ts > visibleEnd) continue;
            const x = timeToX(sync.ts);
            ctx.beginPath();
            ctx.moveTo(x, padding.top);
            ctx.lineTo(x, height - padding.bottom);
            ctx.stroke();
        }

        ctx.fillStyle = '#8b949e';
        ctx.font = '10px -apple-system, sans-serif';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        ctx.fillText(maxSize.toFixed(0), padding.left - 8, padding.top);
        ctx.fillText('0', padding.left - 8, height - padding.bottom);

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

        ctx.fillStyle = '#0d1117';
        ctx.fillRect(0, 0, width, height);

        if (!this.bufferEvents.length) {
            ctx.fillStyle = '#8b949e';
            ctx.font = '14px -apple-system, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('No data', width / 2, height / 2);
            return;
        }

        const lastEvent = this.bufferEvents[this.bufferEvents.length - 1];
        const fates = lastEvent.fates || { used: {}, wasted: {}, filtered: {}, failed: {} };

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
            ctx.fillText('No samples processed', width / 2, height / 2);
            return;
        }

        const padding = { left: 60, right: 20, top: 20, bottom: 40 };
        const barWidth = width - padding.left - padding.right;
        const barHeight = 40;
        const barY = height / 2 - barHeight / 2;

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

        ctx.fillStyle = '#8b949e';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        ctx.fillText(`Total: ${total}`, padding.left - 10, height / 2);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new TraceViewer();
});
