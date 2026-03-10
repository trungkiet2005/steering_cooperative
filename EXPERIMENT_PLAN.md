# Experiment Execution Plan — Multi-Model NeurIPS 2026

## Overview

Run the full experiment pipeline across **5 models** on Kaggle H100. Each notebook session runs **ONE model** to stay within 80 GB VRAM. Total: **5 sessions**.

---

## Model Schedule

| Session | `MODEL_KEY` | Model | Params | VRAM Est. | Priority |
|---|---|---|---|---|---|
| 1 | `qwen-32b` | Qwen2.5-32B-Instruct | 32B | ~20 GB | ★★★ Primary (existing data) |
| 2 | `llama-8b` | Llama-3.1-8B-Instruct | 8B | ~6 GB | ★★★ Fast baseline |
| 3 | `mistral-7b` | Mistral-7B-Instruct-v0.3 | 7B | ~5 GB | ★★ Architecture diversity |
| 4 | `gemma-27b` | Gemma-2-27B-it | 27B | ~17 GB | ★★ Google architecture |
| 5 | `llama-70b` | Llama-3.1-70B-Instruct | 70B | ~42 GB | ★ Largest model |

> [!TIP]
> Start with Session 2 (`llama-8b`) — fastest to run (~1-2h), validates the pipeline before committing to larger models.

---

## Per-Session Checklist

For each session, open `notebooks/run_neurips_experiments.ipynb` on Kaggle:

### Core Experiments (~3-5h for 32B, ~1-2h for 7-8B)

```
Step 1: Clone repo
Step 2: Set MODEL_KEY = '<model-key>'
Step 3: run_all_experiments()
  ├── Phase 1: Calibration → steering_vectors.npz
  ├── Phase 2: Baseline IPD → baseline_results.csv
  ├── Phase 3: Prompt-Coop → prompt_coop_results.csv
  ├── Phase 4: Controls → control_results.csv
  ├── Phase 5: α-sweep (SV/CAA/RepE) → steered_results.csv
  ├── Phase 6: Layer ablation → layer_ablation_results.csv
  ├── Novel A: Strategic layer → novel_a_strategic_layer.csv
  ├── Novel B: Dynamic α → novel_b_dynamic.csv
  ├── Novel C: OCE → novel_c_oce.csv
  └── Novel D: Head importance → head_importance.npy
```

### Benchmarks (~1-2h additional)

```
Step 4: Run benchmarks
  ├── B1: WikiText-2 PPL → wikitext2_ppl.csv
  ├── B2: Semantic invariance → semantic_invariance_test.csv
  ├── B3: Cross-lingual → crosslingual_steering.csv
  └── B4: Scenario dilemmas → scenario_dilemma_test.csv
```

### Critical — FDI Sweep (~2-3h for 32B, ~30min for 7B)

```
Step 5: FDI sweep → fdi_sweep.npy, fdi_sweep.png
  └── This confirms the strategic bottleneck layer for each model
```

### Archive

```
Step 8: Archive → results_<model-key>.tar.gz
  └── Download from Kaggle output
```

---

## Output Structure

After all 5 sessions:

```
steering_outputs/
├── qwen-32b/
│   ├── steering_vectors.npz
│   ├── baseline_results.csv
│   ├── steered_results.csv
│   ├── fdi_sweep.npy          ← peak at Layer 57
│   ├── head_importance.npy
│   ├── wikitext2_ppl.csv
│   ├── crosslingual_steering.csv
│   └── ...
├── llama-8b/
│   ├── fdi_sweep.npy          ← expect peak at ~Layer 27
│   └── ...
├── llama-70b/
│   ├── fdi_sweep.npy          ← expect peak at ~Layer 68
│   └── ...
├── mistral-7b/
│   ├── fdi_sweep.npy          ← expect peak at ~Layer 27
│   └── ...
└── gemma-27b/
    ├── fdi_sweep.npy          ← expect peak at ~Layer 39
    └── ...
```

---

## Key Metrics to Collect Per Model

| Metric | Source File | Paper Table/Figure |
|---|---|---|
| Peak FDI layer ($l^*$) | `fdi_sweep.npy` | Table 2 (model config) |
| Depth ratio ($l^*/L$) | computed | Table 2 |
| Baseline coop rate | `baseline_results.csv` | §5 Results |
| SV coop rate @α=0.2 | `steered_results.csv` | §5, Fig 3 |
| PPL ratio @α=0.2 | `wikitext2_ppl.csv` | §5 |
| Cross-lingual transfer | `crosslingual_steering.csv` | Fig (new) |
| OCE recovery @α=0.5 | `novel_c_oce.csv` | §5.4 |

---

## Cross-Model Comparison (After All Sessions)

After collecting all 5 archives, run this analysis locally:

```python
import numpy as np
import pandas as pd

models = ['qwen-32b', 'llama-8b', 'llama-70b', 'mistral-7b', 'gemma-27b']
n_layers = [64, 32, 80, 32, 46]

# 1. Strategic Bottleneck Comparison
print("Model           | Layers | Peak FDI Layer | Depth Ratio")
print("-" * 55)
for model, nl in zip(models, n_layers):
    fdi = np.load(f"steering_outputs/{model}/fdi_sweep.npy")
    peak = int(np.argmax(fdi))
    ratio = peak / nl
    print(f"{model:16s} | {nl:5d}  | {peak:14d} | {ratio:.3f}")

# 2. Cooperation Rate Comparison
for model in models:
    df = pd.read_csv(f"steering_outputs/{model}/steered_results.csv")
    sv = df[(df['method'] == 'steered') & (df['alpha'] == 0.2)]
    coop = sv['coop_rate'].mean()
    print(f"{model}: SV @α=0.2 → coop = {coop:.1%}")
```

---

## Timeline Estimate

| Session | Model | Est. Runtime | GPU Hours |
|---|---|---|---|
| 1 | `qwen-32b` | 6-8h | 8h |
| 2 | `llama-8b` | 2-3h | 3h |
| 3 | `mistral-7b` | 2-3h | 3h |
| 4 | `gemma-27b` | 5-7h | 7h |
| 5 | `llama-70b` | 8-12h | 12h |
| **Total** | | | **~33h** |

> [!IMPORTANT]
> Kaggle free tier: 30h/week GPU. You may need 2 weeks or a Kaggle Pro account.

---

## Troubleshooting

| Issue | Solution |
|---|---|
| OOM on 70B | Use `SKIP_PHASES = ['phase5']` to skip full α-sweep first |
| Llama gating | Need Hugging Face token: `huggingface-cli login` |
| Slow FDI sweep | Reduce rounds: `run_fdi_sweep(player, cfg, out_dir, n_rounds=10)` |
| Gemma tokenizer | Add `trust_remote_code=True` (already set in config) |
