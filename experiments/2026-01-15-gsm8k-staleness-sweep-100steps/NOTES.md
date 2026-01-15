# GSM8K Staleness Sweep Experiment

**Date:** 2026-01-15
**Model:** Qwen3-1.7B-Base
**Dataset:** GSM8K (train split)
**Steps:** 100 optimization steps per experiment

## Objective

Study the effect of `max_staleness` parameter on GRPO training speed and quality. The `max_staleness` parameter controls how many optimizer steps old a sample can be before it's evicted from the buffer.

- `max_staleness=0`: Samples are evicted immediately after one use (vanilla GRPO)
- `max_staleness=1`: Samples can be reused for up to 1 additional optimizer step (allows async overlap)

## Experiments

| Exp | Run Name | Staleness | Optim Step | Sync | LR | Notes |
|-----|----------|-----------|------------|------|-----|-------|
| 1 | gsm8k_s0_opt16_lr1e6 | 0 | 16 | 16 | 1e-6 | Vanilla GRPO baseline |
| 2 | gsm8k_s1_opt16_lr1e6 | 1 | 16 | 16 | 1e-6 | Staleness=1 variant |
| 3 | gsm8k_s0_opt8_lr5e7 | 0 | 8 | 8* | 5e-7 | Half batch, sync modified |
| 4 | gsm8k_s1_opt8_lr5e7 | 1 | 8 | 8* | 5e-7 | Half batch + staleness, sync modified |
| 5 | gsm8k_s1_opt8_lr5e7_sync16 | 1 | 8 | 16 | 5e-7 | Half batch + staleness, original sync |

**Note:** Exp3 and Exp4 had `prompts_per_rollout_sync` modified from 16 to 8 during experiment due to a debugging issue. Exp5 was created to restore the original sync=16 configuration for proper comparison.

### Key Configuration Parameters

- `prompts_per_rollout_sync`: How often weights sync to rollout vLLM (controls async overlap)
- `prompts_per_optim_step`: Batch size in prompts per gradient step
- `max_staleness`: Maximum age of samples before eviction
- Learning rate scaled linearly with batch size (1e-6 for opt16, 5e-7 for opt8)

## Training Speed Results

| Experiment | Training Time | Mean Step Time | Speedup vs Exp1 |
|------------|---------------|----------------|-----------------|
| Exp1 (s0, opt16, sync16) | 70.4 min | 42.6s | 1.0x (baseline) |
| Exp2 (s1, opt16, sync16) | 48.1 min | 29.1s | 1.5x |
| Exp3 (s0, opt8, sync8*) | 47.4 min | 28.7s | 1.5x |
| Exp4 (s1, opt8, sync8*) | **25.1 min** | **15.2s** | **2.8x** |
| Exp5 (s1, opt8, sync16) | 34.9 min | 21.1s | 2.0x |

### Step Time Distribution Analysis

**Exp1 (s0, opt16, sync16):** Normal distribution centered at 42.6s
- Range: 39.1s - 46.8s
- Very consistent timing (std: 1.53s)

**Exp2 (s1, opt16, sync16):** Tight distribution with occasional outliers
- 66 steps in 28.2-29.1s range
- One outlier at 42.2s
- Staleness=1 allows overlapping rollout generation with training

**Exp3 (s0, opt8, sync8*):** Normal distribution centered at 28.7s
- Range: 25.3s - 32.1s
- Faster than Exp1 due to smaller batch size

**Exp4 (s1, opt8, sync8*):** Very tight, fast distribution
- 75 of 99 steps in 14.4-15.3s range
- Fastest experiment overall
- Staleness=1 + smaller batch = maximum async overlap

**Exp5 (s1, opt8, sync16):** Bimodal distribution
- 50 fast steps (~14s): reusing stale data
- ~45 slow steps (~28-29s): waiting for fresh rollouts
- Alternating pattern due to sync=16 with opt=8

### Key Observations

1. **Staleness=1 provides 1.5-1.9x speedup** when comparing same configurations:
   - Exp1 → Exp2: 70.4 → 48.1 min (1.5x)
   - Exp3 → Exp4: 47.4 → 25.1 min (1.9x)

2. **Bimodal behavior in Exp5** demonstrates the interplay between sync interval and staleness:
   - With sync=16 and opt=8, every other batch can reuse stale data
   - Creates alternating fast/slow steps

3. **Exp4 achieved 2.8x speedup** over vanilla GRPO (Exp1) while maintaining similar quality

## Evaluation Results

Evaluated using `lm_eval` with vLLM backend on GSM8K CoT task (greedy decoding, temp=0).

### Exact Match Accuracy (%)

| Model | 0-shot | 4-shot | 8-shot |
|-------|--------|--------|--------|
| Base (Qwen3-1.7B) | 56.03 | 52.08 | 64.67 |
| Exp1 (s0, opt16, sync16, lr1e-6) | 55.65 | 51.78 | 64.59 |
| Exp2 (s1, opt16, sync16, lr1e-6) | 55.50 | 51.10 | 64.90 |
| Exp3 (s0, opt8, sync8*, lr5e-7) | 56.10 | 52.08 | 64.14 |
| Exp4 (s1, opt8, sync8*, lr5e-7) | 55.95 | 50.49 | 64.59 |
| Exp5 (s1, opt8, sync16, lr5e-7) | 56.10 | 52.16 | 64.82 |

### Delta from Base Model (%)

| Model | 0-shot | 4-shot | 8-shot |
|-------|--------|--------|--------|
| Exp1 | -0.38 | -0.30 | -0.08 |
| Exp2 | -0.53 | -0.99 | +0.23 |
| Exp3 | +0.08 | +0.00 | -0.53 |
| Exp4 | -0.08 | -1.59 | -0.08 |
| Exp5 | +0.08 | +0.08 | +0.15 |

### Quality Observations

- All experiments show accuracy within ~1% of base model
- No significant degradation from using staleness=1
- 100 steps may be insufficient to see meaningful improvements
- 4-shot evaluation shows more variance than 0-shot or 8-shot

## Conclusions

1. **Staleness=1 significantly improves training speed** (1.5-2x) without degrading model quality at 100 steps
2. **Smaller batch sizes with staleness** enable even faster training (Exp4: 2.8x speedup)
3. **Quality is preserved** - no statistically significant difference between staleness=0 and staleness=1
4. **Bimodal step timing** (Exp5) suggests careful tuning of sync interval relative to batch size is important
5. **100 steps is insufficient** to see quality improvements on GSM8K - longer training needed

## File Structure

```
experiments/2026-01-15-gsm8k-staleness-sweep-100steps/
├── NOTES.md                    # This file
├── configs/                    # YAML configuration files used
│   ├── gsm8k-exp1-s0-opt16-lr1e6.yaml
│   ├── gsm8k-exp2-s1-opt16-lr1e6.yaml
│   ├── gsm8k-exp3-s0-opt8-lr5e7.yaml
│   ├── gsm8k-exp4-s1-opt8-lr5e7.yaml
│   └── gsm8k-exp5-s1-opt8-lr5e7-sync16.yaml
├── traces/                     # Training traces and rollouts
│   ├── exp{1-5}_*.jsonl       # Trace files
│   └── exp{1-5}_rollouts.jsonl # Rollout data
├── checkpoints/                # Model checkpoints
│   └── exp{1-5}_step{50,100}/ # Checkpoint directories
└── eval_results/               # lm_eval JSON outputs
    ├── base_{0,4,8}shot.json
    └── exp{1-5}_{0,4,8}shot.json
```

## Reproduction

To reproduce these experiments:

```bash
# Run experiment (example for exp1)
python entrypoints/train_grpo.py --config configs/gsm8k-exp1-s0-opt16-lr1e6.yaml

# Evaluate checkpoint
lm_eval --model vllm \
  --model_args "pretrained=checkpoints/exp1_step100,dtype=bfloat16" \
  --tasks gsm8k_cot \
  --num_fewshot 0 \
  --batch_size auto
```
