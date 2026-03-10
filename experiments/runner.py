"""
Latent Altruism — Main Experiment Runner
=========================================
Orchestrates all experiment phases for a SINGLE model.
Usage:
    python runner.py --model qwen-32b
    python runner.py --model llama-8b
    MODEL_KEY=mistral-7b python runner.py
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

from .config import SteeringConfig, MODEL_REGISTRY, get_model_config
from .games import get_opponent_action, calculate_payoff
from .model import SteeringLLMPlayer, NumpyEncoder
from .steering import (
    compute_fdi, compute_silhouette, compute_cosine_sim,
    aggregate_stats, significance_tests,
)

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 7)
plt.rcParams['font.size'] = 12


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: CALIBRATION (SV extraction)
# ─────────────────────────────────────────────────────────────────────────────

def run_phase1_calibration(player: SteeringLLMPlayer, cfg: SteeringConfig,
                           out_dir: str):
    """Extract AllC/AllD hidden states → compute steering vectors."""
    print(f"\n{'='*60}")
    print(f"PHASE 1: CALIBRATION — {cfg.model_key}")
    print(f"{'='*60}\n")

    layers = cfg.EXTRACTION_LAYERS
    n_rounds = cfg.CALIBRATION_ROUNDS
    opp = cfg.CALIBRATION_OPPONENT

    allc_vecs = player.collect_strategy_vectors('AllC', opp, n_rounds, layers)
    alld_vecs = player.collect_strategy_vectors('AllD', opp, n_rounds, layers)

    # Compute SVs per extraction layer
    svs = {}
    meta = {'model': cfg.model_key, 'layers': {}}
    for li in layers:
        sv = player.compute_steering_vector(allc_vecs[li], alld_vecs[li])
        svs[li] = sv
        norm = float(np.linalg.norm(sv))
        fdi = compute_fdi(allc_vecs[li], alld_vecs[li])
        sil = compute_silhouette(allc_vecs[li], alld_vecs[li])
        meta['layers'][str(li)] = {
            'sv_norm': norm, 'fdi': fdi, 'silhouette': sil,
        }
        print(f"  Layer {li}: ||sv|| = {norm:.4f}, FDI = {fdi:.2f}, "
              f"Silhouette = {sil:.4f}")

    # Save
    np.savez(f"{out_dir}/steering_vectors.npz",
             **{f"sv_{li}": svs[li] for li in layers})
    np.savez(f"{out_dir}/allc_vectors.npz",
             **{f"layer_{li}": np.array(allc_vecs[li]) for li in layers})
    np.savez(f"{out_dir}/alld_vectors.npz",
             **{f"layer_{li}": np.array(alld_vecs[li]) for li in layers})
    with open(f"{out_dir}/calibration_meta.json", 'w') as f:
        json.dump(meta, f, indent=2, cls=NumpyEncoder)

    print(f"\n✓ Phase 1 complete — SVs saved to {out_dir}/")
    return svs, allc_vecs, alld_vecs, meta


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2: BASELINE IPD
# ─────────────────────────────────────────────────────────────────────────────

def run_phase2_baseline(player: SteeringLLMPlayer, cfg: SteeringConfig,
                        out_dir: str):
    """Run unsteered games against all opponents."""
    print(f"\n{'='*60}")
    print(f"PHASE 2: BASELINE IPD")
    print(f"{'='*60}\n")

    results = []
    for opp in cfg.EVAL_OPPONENTS:
        for rep in range(cfg.EVAL_GAMES_PER_OPPONENT):
            r = player.play_game(
                opponent=opp, n_rounds=cfg.EVAL_ROUNDS,
                method='baseline', label='Baseline', verbose=True)
            r['rep'] = rep
            results.append(r)

    df = pd.DataFrame(results)
    df.to_csv(f"{out_dir}/baseline_results.csv", index=False)
    print(f"\n✓ Phase 2 complete — {len(results)} games")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3: PROMPT-COOPERATIVE BASELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_phase3_prompt_coop(player: SteeringLLMPlayer, cfg: SteeringConfig,
                           out_dir: str):
    """Run games with cooperation instruction (no SV)."""
    print(f"\n{'='*60}")
    print(f"PHASE 3: PROMPT-COOPERATIVE BASELINE")
    print(f"{'='*60}\n")

    results = []
    for opp in cfg.EVAL_OPPONENTS:
        for rep in range(cfg.EVAL_GAMES_PER_OPPONENT):
            r = player.play_game(
                opponent=opp, n_rounds=cfg.EVAL_ROUNDS,
                instruction=cfg.COOPERATE_INSTRUCTION,
                method='baseline', label='PromptCoop', verbose=True)
            r['rep'] = rep
            results.append(r)

    df = pd.DataFrame(results)
    df.to_csv(f"{out_dir}/prompt_coop_results.csv", index=False)
    print(f"\n✓ Phase 3 complete")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4: CONTROL VECTORS
# ─────────────────────────────────────────────────────────────────────────────

def run_phase4_controls(player: SteeringLLMPlayer, cfg: SteeringConfig,
                        sv: np.ndarray, out_dir: str):
    """Random and orthogonal control vectors as negative controls."""
    print(f"\n{'='*60}")
    print(f"PHASE 4: CONTROL VECTORS")
    print(f"{'='*60}\n")

    dim = sv.shape[0]
    results = []

    for ctrl_type in ['Random', 'Orthogonal']:
        if ctrl_type == 'Random':
            ctrl_sv = np.random.randn(dim).astype(np.float32)
            ctrl_sv = ctrl_sv / np.linalg.norm(ctrl_sv) * np.linalg.norm(sv)
        else:
            # Gram-Schmidt orthogonalize
            ctrl_sv = np.random.randn(dim).astype(np.float32)
            ctrl_sv -= (np.dot(ctrl_sv, sv) / np.dot(sv, sv)) * sv
            ctrl_sv = ctrl_sv / np.linalg.norm(ctrl_sv) * np.linalg.norm(sv)

        for alpha in [0.1, 0.2, 0.3, 0.5]:
            for opp in ['TFT']:
                for rep in range(cfg.EVAL_GAMES_PER_OPPONENT):
                    r = player.play_game(
                        opponent=opp, n_rounds=cfg.EVAL_ROUNDS,
                        sv=ctrl_sv, alpha=alpha,
                        method='steered',
                        label=f'Control_{ctrl_type}',
                        verbose=True)
                    r['rep'] = rep
                    results.append(r)

    df = pd.DataFrame(results)
    df.to_csv(f"{out_dir}/control_results.csv", index=False)
    print(f"\n✓ Phase 4 complete")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5: STEERED GAMES — Full α sweep, SV vs CAA vs RepE
# ─────────────────────────────────────────────────────────────────────────────

def run_phase5_steered(player: SteeringLLMPlayer, cfg: SteeringConfig,
                       svs: dict, out_dir: str):
    """Full α sweep with three steering methods."""
    print(f"\n{'='*60}")
    print(f"PHASE 5: STEERED GAMES — α SWEEP")
    print(f"{'='*60}\n")

    sv = svs[cfg.PRIMARY_LAYER]
    results = []

    for method in ['steered', 'caa', 'repe']:
        for alpha in cfg.ALPHA_SWEEP:
            for opp in cfg.EVAL_OPPONENTS:
                for rep in range(cfg.EVAL_GAMES_PER_OPPONENT):
                    r = player.play_game(
                        opponent=opp, n_rounds=cfg.EVAL_ROUNDS,
                        sv=sv, alpha=alpha,
                        method=method,
                        label=f'{method.upper()}_a{alpha}',
                        verbose=True)
                    r['rep'] = rep
                    results.append(r)
            torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    df.to_csv(f"{out_dir}/steered_results.csv", index=False)
    print(f"\n✓ Phase 5 complete — {len(results)} games")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 6: LAYER ABLATION
# ─────────────────────────────────────────────────────────────────────────────

def run_phase6_layer_ablation(player: SteeringLLMPlayer, cfg: SteeringConfig,
                              svs: dict, out_dir: str):
    """Compare SVs extracted from different layers."""
    print(f"\n{'='*60}")
    print(f"PHASE 6: LAYER ABLATION")
    print(f"{'='*60}\n")

    results = []
    for li in cfg.EXTRACTION_LAYERS:
        if li not in svs:
            continue
        sv = svs[li]
        for alpha in [0.1, 0.2, 0.3]:
            for opp in ['TFT']:
                for rep in range(cfg.EVAL_GAMES_PER_OPPONENT):
                    r = player.play_game(
                        opponent=opp, n_rounds=cfg.EVAL_ROUNDS,
                        sv=sv, alpha=alpha,
                        method='steered',
                        label=f'Layer{li}_a{alpha}',
                        verbose=True)
                    r['rep'] = rep
                    results.append(r)

    df = pd.DataFrame(results)
    df.to_csv(f"{out_dir}/layer_ablation_results.csv", index=False)
    print(f"\n✓ Phase 6 complete")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# NOVEL EXP A: LAYER 57 TARGETED INJECTION
# ─────────────────────────────────────────────────────────────────────────────

def run_novel_a_strategic_layer(player: SteeringLLMPlayer, cfg: SteeringConfig,
                                 sv: np.ndarray, out_dir: str):
    """Compare injection at strategic layer vs last layer."""
    print(f"\n{'='*60}")
    print(f"NOVEL A: STRATEGIC LAYER {cfg.STRATEGIC_LAYER} vs LAST LAYER")
    print(f"{'='*60}\n")

    results = []
    for alpha in [0.05, 0.1, 0.2, 0.3, 0.5]:
        for opp in ['TFT', 'AllD']:
            for rep in range(cfg.EVAL_GAMES_PER_OPPONENT):
                # Last layer (existing method)
                r1 = player.play_game(
                    opponent=opp, n_rounds=cfg.EVAL_ROUNDS,
                    sv=sv, alpha=alpha,
                    method='steered', label=f'LastLayer_a{alpha}',
                    verbose=True)
                r1['rep'] = rep
                results.append(r1)

                # Strategic layer
                r2 = player.play_game(
                    opponent=opp, n_rounds=cfg.EVAL_ROUNDS,
                    sv=sv, alpha=alpha,
                    method='steered_at_layer',
                    target_layer=cfg.STRATEGIC_LAYER,
                    label=f'Layer{cfg.STRATEGIC_LAYER}_a{alpha}',
                    verbose=True)
                r2['rep'] = rep
                results.append(r2)

    df = pd.DataFrame(results)
    df.to_csv(f"{out_dir}/novel_a_strategic_layer.csv", index=False)
    print(f"\n✓ Novel A complete")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# NOVEL EXP B: DYNAMIC α STEERING (Latent TfT)
# ─────────────────────────────────────────────────────────────────────────────

def run_novel_b_dynamic(player: SteeringLLMPlayer, cfg: SteeringConfig,
                        sv: np.ndarray, out_dir: str):
    """Dynamic α steering that mimics TfT at the activation level."""
    print(f"\n{'='*60}")
    print(f"NOVEL B: DYNAMIC α STEERING")
    print(f"{'='*60}\n")

    results = []
    for opp in cfg.EVAL_OPPONENTS:
        for rep in range(cfg.EVAL_GAMES_PER_OPPONENT):
            r = player.play_game(
                opponent=opp, n_rounds=cfg.EVAL_ROUNDS,
                sv=sv, alpha=cfg.DYN_ALPHA_HIGH,
                method='dynamic', label='Dynamic_TfT',
                verbose=True)
            r['rep'] = rep
            results.append(r)

    df = pd.DataFrame(results)
    df.to_csv(f"{out_dir}/novel_b_dynamic.csv", index=False)
    print(f"\n✓ Novel B complete")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# NOVEL EXP C: ORTHOGONAL CONCEPT ERASURE (OCE)
# ─────────────────────────────────────────────────────────────────────────────

def run_novel_c_oce(player: SteeringLLMPlayer, cfg: SteeringConfig,
                    sv: np.ndarray, out_dir: str):
    """OCE adversarial robustness experiment."""
    print(f"\n{'='*60}")
    print(f"NOVEL C: ORTHOGONAL CONCEPT ERASURE")
    print(f"{'='*60}\n")

    # Extract adversarial direction
    adv_sv = player.compute_adversarial_vector(
        cfg.ADV_PROMPTS_NEUTRAL, cfg.ADV_PROMPTS_HOSTILE,
        cfg.ADV_COLLECTION_ROUNDS)

    cos_sim = compute_cosine_sim(sv, adv_sv)
    print(f"  cos(v_coop, v_adv) = {cos_sim:.4f}")

    results = []
    for adv_prompt in cfg.ADV_PROMPTS_HOSTILE:
        adv_name = adv_prompt.strip()[:30]
        for alpha in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]:
            for rep in range(cfg.EVAL_GAMES_PER_OPPONENT):
                # Adversarial only
                r1 = player.play_game(
                    opponent='TFT', n_rounds=cfg.EVAL_ROUNDS,
                    instruction=adv_prompt,
                    method='baseline', label='Adv_NoSV',
                    verbose=True)
                r1['rep'] = rep
                r1['adv_type'] = adv_name
                results.append(r1)

                # Adversarial + last-layer SV
                r2 = player.play_game(
                    opponent='TFT', n_rounds=cfg.EVAL_ROUNDS,
                    sv=sv, alpha=alpha,
                    instruction=adv_prompt,
                    method='steered', label=f'Adv+SV_a{alpha}',
                    verbose=True)
                r2['rep'] = rep
                r2['adv_type'] = adv_name
                results.append(r2)

                # OCE
                if alpha > 0:
                    r3 = player.play_game(
                        opponent='TFT', n_rounds=cfg.EVAL_ROUNDS,
                        sv=sv, adv_sv=adv_sv, alpha=alpha,
                        instruction=adv_prompt,
                        method='erasure', label=f'OCE_a{alpha}',
                        verbose=True)
                    r3['rep'] = rep
                    r3['adv_type'] = adv_name
                    results.append(r3)

    df = pd.DataFrame(results)
    df.to_csv(f"{out_dir}/novel_c_oce.csv", index=False)
    np.save(f"{out_dir}/adversarial_vector.npy", adv_sv)

    print(f"\n✓ Novel C complete — {len(results)} games")
    return df, adv_sv


# ─────────────────────────────────────────────────────────────────────────────
# NOVEL EXP D: ATTENTION HEAD IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

def run_novel_d_head_importance(player: SteeringLLMPlayer, cfg: SteeringConfig,
                                 out_dir: str):
    """Per-head importance analysis at the strategic layer."""
    print(f"\n{'='*60}")
    print(f"NOVEL D: HEAD IMPORTANCE AT LAYER {cfg.STRATEGIC_LAYER}")
    print(f"{'='*60}\n")

    allc_vecs_57 = player.collect_layer57_vectors(
        cfg.STRATEGY_PROMPTS['AllC'], cfg.CALIBRATION_ROUNDS,
        cfg.STRATEGIC_LAYER)
    alld_vecs_57 = player.collect_layer57_vectors(
        cfg.STRATEGY_PROMPTS['AllD'], cfg.CALIBRATION_ROUNDS,
        cfg.STRATEGIC_LAYER)

    head_imp = player.compute_head_importance(
        allc_vecs_57, alld_vecs_57, cfg.STRATEGIC_LAYER)

    # Save
    np.save(f"{out_dir}/head_importance.npy", head_imp)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 5))
    top5 = np.argsort(head_imp)[-5:]
    colors = ['#e74c3c' if i in top5 else '#3498db'
              for i in range(len(head_imp))]
    ax.bar(range(len(head_imp)), head_imp, color=colors)
    ax.set_xlabel('Attention Head Index')
    ax.set_ylabel('||Δh|| (AllC − AllD)')
    ax.set_title(f'Per-Head Strategic Importance — Layer {cfg.STRATEGIC_LAYER}'
                 f' ({cfg.model_key})')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/head_importance.png", dpi=150)
    plt.close()

    print(f"\n✓ Novel D complete — Top 5 heads: {top5}")
    return head_imp


# ─────────────────────────────────────────────────────────────────────────────
# LAYER-WISE FDI SWEEP
# ─────────────────────────────────────────────────────────────────────────────

def run_fdi_sweep(player: SteeringLLMPlayer, cfg: SteeringConfig,
                  out_dir: str, n_rounds: int = 15):
    """Compute FDI at every transformer layer — finds strategic bottleneck."""
    print(f"\n{'='*60}")
    print(f"FDI SWEEP — {cfg.N_LAYERS} layers")
    print(f"{'='*60}\n")

    fdi_values = []
    # Scan all layers
    for layer_idx in range(cfg.N_LAYERS):
        allc_v = player.collect_layer57_vectors(
            cfg.STRATEGY_PROMPTS['AllC'], n_rounds, layer_idx)
        alld_v = player.collect_layer57_vectors(
            cfg.STRATEGY_PROMPTS['AllD'], n_rounds, layer_idx)
        fdi = compute_fdi(allc_v, alld_v)
        fdi_values.append(fdi)
        print(f"  Layer {layer_idx}: FDI = {fdi:.2f}")
        torch.cuda.empty_cache()

    peak_layer = int(np.argmax(fdi_values))
    print(f"\n  ★ Peak FDI: Layer {peak_layer} = {fdi_values[peak_layer]:.2f}")

    # Save
    np.save(f"{out_dir}/fdi_sweep.npy", np.array(fdi_values))

    # Plot
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(range(cfg.N_LAYERS), fdi_values, 'b-', linewidth=2)
    ax.axvline(peak_layer, color='r', linestyle='--', label=f'Peak: L{peak_layer}')
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Fisher Discriminability Index')
    ax.set_title(f'Layer-wise FDI — {cfg.model_key}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/fdi_sweep.png", dpi=150)
    plt.close()

    return fdi_values, peak_layer


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def run_all_experiments(model_key: str = None,
                        skip_phases: list = None,
                        output_root: str = None):
    """Run the complete experiment pipeline for ONE model.

    Args:
        model_key:   Key from MODEL_REGISTRY (e.g. 'qwen-32b', 'llama-8b')
        skip_phases: List of phase names to skip (e.g. ['phase4', 'novel_d'])
        output_root: Root output directory (default: /kaggle/working/steering_outputs)
    """
    skip = set(skip_phases or [])

    # ── Setup ──
    cfg = SteeringConfig(model_key)
    if output_root:
        cfg.OUTPUT_DIR = f"{output_root}/{cfg.model_key}"
    out_dir = cfg.OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    cfg.print_summary()

    # ── Load model ──
    player = SteeringLLMPlayer(cfg)

    # ── Phase 1: Calibration ──
    svs, allc_vecs, alld_vecs, cal_meta = run_phase1_calibration(
        player, cfg, out_dir)
    sv = svs[cfg.PRIMARY_LAYER]

    # ── Phase 2: Baseline ──
    if 'phase2' not in skip:
        run_phase2_baseline(player, cfg, out_dir)

    # ── Phase 3: Prompt Coop ──
    if 'phase3' not in skip:
        run_phase3_prompt_coop(player, cfg, out_dir)

    # ── Phase 4: Controls ──
    if 'phase4' not in skip:
        run_phase4_controls(player, cfg, sv, out_dir)

    # ── Phase 5: Steered ──
    if 'phase5' not in skip:
        run_phase5_steered(player, cfg, svs, out_dir)

    # ── Phase 6: Layer ablation ──
    if 'phase6' not in skip:
        run_phase6_layer_ablation(player, cfg, svs, out_dir)

    # ── Novel A: Strategic layer ──
    if 'novel_a' not in skip:
        run_novel_a_strategic_layer(player, cfg, sv, out_dir)

    # ── Novel B: Dynamic ──
    if 'novel_b' not in skip:
        run_novel_b_dynamic(player, cfg, sv, out_dir)

    # ── Novel C: OCE ──
    if 'novel_c' not in skip:
        run_novel_c_oce(player, cfg, sv, out_dir)

    # ── Novel D: Head importance ──
    if 'novel_d' not in skip:
        run_novel_d_head_importance(player, cfg, out_dir)

    print(f"\n{'='*60}")
    print(f"ALL EXPERIMENTS COMPLETE — {cfg.model_key}")
    print(f"Results saved to: {out_dir}")
    print(f"{'='*60}\n")

    return player, cfg, svs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Latent Altruism experiments for a single model")
    parser.add_argument(
        '--model', type=str, default=None,
        choices=list(MODEL_REGISTRY.keys()),
        help=f"Model key: {list(MODEL_REGISTRY.keys())}")
    parser.add_argument(
        '--skip', nargs='*', default=[],
        help="Phases to skip (e.g. phase4 novel_d)")
    parser.add_argument(
        '--output', type=str, default=None,
        help="Output root directory")
    args = parser.parse_args()

    if torch.cuda.is_available():
        print(f"GPU  : {torch.cuda.get_device_name(0)}")
        print(f"VRAM : {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        print("WARNING: No GPU detected — running on CPU (very slow)")

    run_all_experiments(
        model_key=args.model,
        skip_phases=args.skip,
        output_root=args.output,
    )
