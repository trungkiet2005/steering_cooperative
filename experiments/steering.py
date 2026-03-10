"""
Latent Altruism — Steering & Statistical Analysis
===================================================
Steering vector math, Fisher Discriminability Index, statistical tests,
and aggregation utilities.
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from typing import List, Dict


# ─────────────────────────────────────────────────────────────────────────────
# FISHER DISCRIMINABILITY INDEX (FDI)
# ─────────────────────────────────────────────────────────────────────────────

def compute_fdi(allc_vecs: List[np.ndarray],
                alld_vecs: List[np.ndarray]) -> float:
    """Compute layer-wise Fisher Discriminability Index.

    FDI = ||μ_C - μ_D||² / σ²_pooled
    where σ²_pooled = mean of per-dim pooled variances
    """
    allc = np.array(allc_vecs)
    alld = np.array(alld_vecs)

    mu_c = allc.mean(axis=0)
    mu_d = alld.mean(axis=0)

    var_c = allc.var(axis=0, ddof=1)
    var_d = alld.var(axis=0, ddof=1)

    pooled_var = (var_c + var_d) / 2
    mean_var = pooled_var.mean()

    if mean_var < 1e-12:
        return 0.0

    delta_sq = np.sum((mu_c - mu_d) ** 2)
    return float(delta_sq / mean_var)


def compute_silhouette(allc_vecs: List[np.ndarray],
                       alld_vecs: List[np.ndarray]) -> float:
    """Simplified silhouette score for two clusters."""
    allc = np.array(allc_vecs)
    alld = np.array(alld_vecs)

    mu_c = allc.mean(axis=0)
    mu_d = alld.mean(axis=0)

    # Intra-cluster: mean dist to own centroid
    a_c = np.mean(np.linalg.norm(allc - mu_c, axis=1))
    a_d = np.mean(np.linalg.norm(alld - mu_d, axis=1))

    # Inter-cluster: mean dist to other centroid
    b_c = np.mean(np.linalg.norm(allc - mu_d, axis=1))
    b_d = np.mean(np.linalg.norm(alld - mu_c, axis=1))

    sil_c = (b_c - a_c) / max(a_c, b_c)
    sil_d = (b_d - a_d) / max(a_d, b_d)

    return float((sil_c + sil_d) / 2)


def compute_cosine_sim(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICAL AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean, std, and 95% CI for cooperation and payoff."""
    rows = []
    for (cond, alpha, opp), grp in df.groupby(
            ['condition', 'alpha', 'opponent']):
        cr = grp['coop_rate'].values
        pay = grp['avg_payoff'].values
        n = len(cr)
        cr_m = cr.mean()
        cr_s = cr.std(ddof=1) if n > 1 else 0.0
        pay_m = pay.mean()
        pay_s = pay.std(ddof=1) if n > 1 else 0.0
        if n > 1:
            t = scipy_stats.t.ppf(0.975, df=n - 1)
            cr_ci = t * cr_s / np.sqrt(n)
            pay_ci = t * pay_s / np.sqrt(n)
        else:
            cr_ci = pay_ci = 0.0
        rows.append({
            'condition': cond, 'alpha': alpha, 'opponent': opp, 'n': n,
            'coop_mean': cr_m, 'coop_std': cr_s, 'coop_ci95': cr_ci,
            'payoff_mean': pay_m, 'payoff_std': pay_s, 'payoff_ci95': pay_ci,
        })
    return pd.DataFrame(rows)


def significance_tests(df: pd.DataFrame,
                       baseline_cond: str = 'Baseline') -> pd.DataFrame:
    """Wilcoxon signed-rank tests + effect sizes vs baseline."""
    rows = []
    bl_df = df[df['condition'] == baseline_cond]
    for (cond, alpha, opp), grp in df[df['condition'] != baseline_cond] \
            .groupby(['condition', 'alpha', 'opponent']):
        bl = bl_df[bl_df['opponent'] == opp]['coop_rate'].values
        st = grp['coop_rate'].values
        n = min(len(bl), len(st))
        if n < 3:
            p, d = np.nan, np.nan
        else:
            bl_s, st_s = bl[:n], st[:n]
            diff = st_s - bl_s
            if np.all(diff == 0):
                p, d = 1.0, 0.0
            else:
                try:
                    _, p = scipy_stats.wilcoxon(bl_s, st_s)
                except ValueError:
                    p = np.nan
                pooled = np.sqrt((bl_s.var(ddof=1) + st_s.var(ddof=1)) / 2)
                d = abs(st_s.mean() - bl_s.mean()) / pooled if pooled > 0 else 0
        rows.append({
            'condition': cond, 'alpha': alpha, 'opponent': opp,
            'n': n, 'p_value': p, 'cohens_d': d,
        })
    return pd.DataFrame(rows)
