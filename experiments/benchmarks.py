"""
Latent Altruism — Benchmarks Module
=====================================
Reviewer response experiments:
  B1: WikiText-2 perplexity (FIXED: steering applied at ALL positions)
  B2: Semantic invariance (ENHANCED: more rounds, temperature decoding, C/D scheme)
  B3: Cross-lingual transfer (FIXED: AllD opponent added, Korean added)
  B4: Scenario-based social dilemmas
  B5: Game transfer (NEW: Stag Hunt + Chicken Game)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from typing import List, Tuple, Dict, Optional

from .config import SteeringConfig
from .games import get_opponent_action, calculate_payoff, GAME_MATRICES
from .model import SteeringLLMPlayer


# ─────────────────────────────────────────────────────────────────────────────
# B1: STANDARD PERPLEXITY BENCHMARK (WikiText-2)
# FIXED: Steering applied to ALL token positions (not just last token)
# ─────────────────────────────────────────────────────────────────────────────

def compute_perplexity_standard(player: SteeringLLMPlayer,
                                cfg: SteeringConfig,
                                sv: np.ndarray,
                                out_dir: str,
                                alphas: List[float] = None,
                                max_samples: int = 200):
    """Compute PPL on WikiText-2 with steering applied to ALL token positions.

    Previous bug: steering was only applied to the last token position,
    so PPL was identical across all α values. Now we inject α·sv into
    every token position (CAA-style), which is the correct way to measure
    whether steering degrades general language capability.
    """
    print(f"\n{'='*60}")
    print("B1: WikiText-2 Perplexity Benchmark")
    print(f"{'='*60}\n")

    from datasets import load_dataset
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    texts = [t for t in dataset['text'] if len(t.strip()) > 50][:max_samples]
    print(f"  Using {len(texts)} WikiText-2 passages")

    if alphas is None:
        alphas = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]

    sv_t = torch.tensor(sv, dtype=torch.float16, device=player.device)
    results = []

    for alpha in alphas:
        total_nll = 0.0
        total_tokens = 0

        for text in texts:
            inputs = player._encode(text)
            seq_len = inputs['input_ids'].shape[1]
            if seq_len < 2:
                continue

            with torch.no_grad():
                if alpha == 0.0:
                    outputs = player.model(**inputs)
                    logits = outputs.logits
                else:
                    # FIX: Apply steering to ALL token positions (CAA-style)
                    outputs = player.model(**inputs, output_hidden_states=True)
                    h = outputs.hidden_states[-1].clone()
                    # Broadcast sv across all positions: h is [B, T, D]
                    h += alpha * sv_t.to(h.dtype)
                    logits = player.model.lm_head(h)

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = inputs['input_ids'][:, 1:].contiguous()

                loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
                nll = loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                total_nll += nll.item()
                total_tokens += shift_labels.numel()

        ppl = float(np.exp(total_nll / total_tokens)) if total_tokens > 0 else float('inf')
        results.append({'alpha': alpha, 'ppl': ppl, 'n_tokens': total_tokens})
        print(f"  α={alpha}: PPL = {ppl:.2f} ({total_tokens} tokens)")
        torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    df.to_csv(f"{out_dir}/wikitext2_ppl.csv", index=False)

    # Plot PPL vs α
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df['alpha'], df['ppl'], 'o-', color='#E53935', linewidth=2,
            markersize=8, label='WikiText-2 PPL')
    if len(df) > 0:
        baseline_ppl = df.loc[df['alpha'] == 0.0, 'ppl'].values
        if len(baseline_ppl) > 0:
            ax.axhline(baseline_ppl[0], color='grey', linestyle='--', alpha=0.5,
                       label=f'Baseline PPL = {baseline_ppl[0]:.2f}')
            # Add ±10% band
            ax.axhspan(baseline_ppl[0] * 0.9, baseline_ppl[0] * 1.1,
                       alpha=0.1, color='green', label='±10% band')
    ax.set_xlabel('Steering α', fontsize=13)
    ax.set_ylabel('Perplexity', fontsize=13)
    ax.set_title('WikiText-2 Perplexity Under Steering', fontsize=14,
                 fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/wikitext2_ppl.png", dpi=150)
    plt.close()

    print(f"\n  ✓ PPL results saved to {out_dir}/wikitext2_ppl.csv")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# B2: SEMANTIC INVARIANCE TEST
# ENHANCED: 50 rounds, temperature decoding, both X/Y and A/B schemes
# ─────────────────────────────────────────────────────────────────────────────

LABEL_SCHEMES = {
    'X/Y': {
        'coop_label': 'X', 'defect_label': 'Y',
        'coop_desc': 'mutual benefit', 'defect_desc': 'self-interest',
    },
    'A/B': {
        'coop_label': 'A', 'defect_label': 'B',
        'coop_desc': 'shared reward', 'defect_desc': 'personal gain',
    },
}


def run_semantic_invariance_test(player: SteeringLLMPlayer,
                                 cfg: SteeringConfig,
                                 sv: np.ndarray,
                                 out_dir: str,
                                 n_rounds: int = 50,
                                 n_reps: int = 3):
    """Test with relabeled actions to rule out lexical priming.

    ENHANCED: Uses 50 rounds (not 10), temperature-based decoding,
    and tests both X/Y and A/B label schemes.
    """
    print(f"\n{'='*60}")
    print("B2: Semantic Invariance Test (X/Y and A/B labels)")
    print(f"{'='*60}\n")

    alphas = [0.0, 0.1, 0.2, 0.3, 0.5]
    sv_t = torch.tensor(sv, dtype=torch.float16, device=player.device)
    results = []

    for scheme_name, scheme in LABEL_SCHEMES.items():
        cl = scheme['coop_label']
        dl = scheme['defect_label']

        prompt_intro = (
            f"You are playing a repeated decision game. In each round:\n"
            f"- Choose Option {cl} ({scheme['coop_desc']}) or "
            f"Option {dl} ({scheme['defect_desc']})\n"
            f"- Payoffs: Both {cl} = 3 pts, Both {dl} = 1 pt, "
            f"{cl} vs {dl} = 0 pts for {cl} and 5 pts for {dl}\n"
        )

        for alpha in alphas:
            for rep in range(n_reps):
                history = []
                opp_last = None
                actions = []

                for rnd in range(n_rounds):
                    prompt = prompt_intro
                    if history:
                        prompt += "Previous rounds:\n"
                        for i, (m, o) in enumerate(history[-5:]):
                            prompt += f"Round {i+1}: You={m}, Other={o}\n"
                    if opp_last:
                        prompt += f"\nOther player's last choice: {opp_last}\n"
                    prompt += f"\nYour choice (respond with only {cl} or {dl}): "

                    inputs = player._encode(prompt)
                    with torch.no_grad():
                        if alpha == 0.0:
                            out = player.model(**inputs)
                            logits_last = out.logits[0, -1, :]
                        else:
                            out = player.model(**inputs, output_hidden_states=True)
                            h = out.hidden_states[-1].clone()
                            h[:, -1, :] += alpha * sv_t.to(h.dtype)
                            logits_last = player.model.lm_head(h)[0, -1, :]

                        # Temperature-based decoding for coop/defect labels
                        c_id = player.tokenizer.encode(cl, add_special_tokens=False)[0]
                        d_id = player.tokenizer.encode(dl, add_special_tokens=False)[0]
                        c_l = logits_last[c_id].item()
                        d_l = logits_last[d_id].item()

                        if cfg.TEMPERATURE <= 0:
                            action = cl if c_l >= d_l else dl
                        else:
                            rl = torch.tensor([c_l, d_l], device=player.device)
                            probs = torch.softmax(rl / cfg.TEMPERATURE, dim=0)
                            action = cl if torch.multinomial(probs, 1).item() == 0 else dl

                    actions.append(action)
                    # Map to C/D for opponent simulation
                    mapped = 'C' if action == cl else 'D'
                    opp_mapped = get_opponent_action('TFT', history, mapped)
                    opp_display = cl if opp_mapped == 'C' else dl
                    history.append((action, opp_display))
                    opp_last = opp_display

                coop_rate = sum(1 for a in actions if a == cl) / len(actions)
                results.append({
                    'alpha': alpha, 'rep': rep, 'coop_rate': coop_rate,
                    'label_scheme': scheme_name,
                })
                print(f"  [{scheme_name}] α={alpha} rep{rep}: "
                      f"{cl}_rate={coop_rate:.1%}")
                torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    df.to_csv(f"{out_dir}/semantic_invariance_test.csv", index=False)

    # Plot: compare C/D rate across schemes
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax_i, scheme_name in enumerate(LABEL_SCHEMES):
        cl = LABEL_SCHEMES[scheme_name]['coop_label']
        sub = df[df['label_scheme'] == scheme_name]
        agg = sub.groupby('alpha')['coop_rate'].agg(['mean', 'std']).reset_index()
        axes[ax_i].errorbar(agg['alpha'], agg['mean'], yerr=agg['std'],
                            fmt='o-', color='#1E88E5', linewidth=2, capsize=5)
        axes[ax_i].set_xlabel('α', fontsize=13)
        axes[ax_i].set_ylabel(f'{cl} Rate (Cooperation)', fontsize=13)
        axes[ax_i].set_title(f'Labels: {scheme_name}', fontsize=13,
                             fontweight='bold')
        axes[ax_i].set_ylim(-0.05, 1.05)
        axes[ax_i].axhline(0.5, color='grey', linestyle='--', alpha=0.3)
    plt.suptitle('Semantic Invariance: Steering Under Label Obfuscation',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/semantic_invariance_test.png", dpi=150)
    plt.close()

    print(f"\n  ✓ Semantic invariance test saved")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# B3: CROSS-LINGUAL STEERING
# FIXED: Test with BOTH TFT and AllD opponents, added Korean
# ─────────────────────────────────────────────────────────────────────────────

LANG_PROMPTS = {
    'en': {
        'intro': ("You are playing the Iterated Prisoner's Dilemma game. "
                  "In each round:\n- Choose C (Cooperate) or D (Defect)\n"
                  "- Payoffs: Both C = 3 pts, Both D = 1 pt, "
                  "C vs D = 0 pts for C and 5 pts for D\n"),
        'history': "Game History (last 5 rounds):\n",
        'round': "Round {i}: You={m}, Opponent={o}\n",
        'opp': "\nOpponent's last move: {opp}\n",
        'ask': "\nYour move (respond with only C or D): ",
    },
    'vi': {
        'intro': ("Bạn đang chơi trò Tình huống Tù nhân lặp lại. "
                  "Mỗi lượt:\n- Chọn C (Hợp tác) hoặc D (Phản bội)\n"
                  "- Điểm: Cả hai C = 3 điểm, Cả hai D = 1 điểm, "
                  "C vs D = 0 điểm cho C và 5 điểm cho D\n"),
        'history': "Lịch sử trò chơi (5 lượt gần nhất):\n",
        'round': "Lượt {i}: Bạn={m}, Đối thủ={o}\n",
        'opp': "\nNước đi gần nhất của đối thủ: {opp}\n",
        'ask': "\nNước đi của bạn (chỉ trả lời C hoặc D): ",
    },
    'zh': {
        'intro': ("你正在玩迭代囚徒困境游戏。"
                  "每一轮：\n- 选择 C（合作）或 D（背叛）\n"
                  "- 得分：双方C = 3分，双方D = 1分，"
                  "C对D = C得0分、D得5分\n"),
        'history': "游戏历史（最近5轮）：\n",
        'round': "第{i}轮：你={m}，对手={o}\n",
        'opp': "\n对手上一轮的选择：{opp}\n",
        'ask': "\n你的选择（仅回答C或D）：",
    },
    'ja': {
        'intro': ("あなたは繰り返し囚人のジレンマゲームをプレイしています。"
                  "各ラウンド：\n- C（協力）またはD（裏切り）を選択\n"
                  "- 得点：両方C = 3点、両方D = 1点、"
                  "C対D = Cは0点、Dは5点\n"),
        'history': "ゲーム履歴（直近5ラウンド）：\n",
        'round': "ラウンド{i}：あなた={m}、相手={o}\n",
        'opp': "\n相手の前回の手：{opp}\n",
        'ask': "\nあなたの手（CまたはDのみで回答）：",
    },
    'ko': {
        'intro': ("당신은 반복 죄수의 딜레마 게임을 하고 있습니다. "
                  "매 라운드:\n- C(협력) 또는 D(배신)를 선택\n"
                  "- 점수: 둘 다 C = 3점, 둘 다 D = 1점, "
                  "C vs D = C는 0점, D는 5점\n"),
        'history': "게임 기록 (최근 5라운드):\n",
        'round': "라운드 {i}: 당신={m}, 상대={o}\n",
        'opp': "\n상대의 마지막 수: {opp}\n",
        'ask': "\n당신의 수 (C 또는 D만 답하세요): ",
    },
}


def run_crosslingual_test(player: SteeringLLMPlayer,
                          cfg: SteeringConfig,
                          sv: np.ndarray,
                          out_dir: str,
                          n_rounds: int = 30,
                          n_reps: int = 3):
    """Test cross-lingual steering transfer.

    FIXED: Tests with BOTH TFT and AllD opponents. AllD is critical
    because some languages (zh, ja) show 100% cooperation at baseline
    with TFT, masking the steering effect. AllD provides a harder test.
    Added Korean language.
    """
    print(f"\n{'='*60}")
    print("B3: Cross-Lingual Steering Transfer")
    print(f"{'='*60}\n")

    alphas = [0.0, 0.1, 0.2, 0.3, 0.5]
    opponents = ['TFT', 'AllD']
    sv_t = torch.tensor(sv, dtype=torch.float16, device=player.device)
    results = []

    for lang in LANG_PROMPTS:
        t = LANG_PROMPTS[lang]
        for opp_strategy in opponents:
            for alpha in alphas:
                for rep in range(n_reps):
                    history = []
                    opp_last = None
                    actions = []

                    for rnd in range(n_rounds):
                        prompt = t['intro'] + "\n"
                        if history:
                            prompt += t['history']
                            window = history[-5:]
                            start = len(history) - len(window) + 1
                            for i, (m, o) in enumerate(window):
                                prompt += t['round'].format(
                                    i=start+i, m=m, o=o)
                        if opp_last:
                            prompt += t['opp'].format(opp=opp_last)
                        prompt += t['ask']

                        inputs = player._encode(prompt)
                        with torch.no_grad():
                            if alpha == 0.0:
                                out = player.model(**inputs)
                                action = player._decode_action(out.logits[0, -1, :])
                            else:
                                out = player.model(**inputs, output_hidden_states=True)
                                h = out.hidden_states[-1].clone()
                                h[:, -1, :] += alpha * sv_t.to(h.dtype)
                                logits = player.model.lm_head(h)
                                action = player._decode_action(logits[0, -1, :])

                        actions.append(action)
                        opp_action = get_opponent_action(opp_strategy, history, action)
                        history.append((action, opp_action))
                        opp_last = opp_action

                    coop = sum(1 for a in actions if a == 'C') / len(actions)
                    results.append({
                        'language': lang, 'opponent': opp_strategy,
                        'alpha': alpha, 'rep': rep, 'coop_rate': coop,
                    })
                    print(f"  [{lang}|{opp_strategy}] α={alpha} rep{rep}: "
                          f"coop={coop:.1%}")
                    torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    df.to_csv(f"{out_dir}/crosslingual_steering.csv", index=False)

    # Plot: two subplots — TFT and AllD
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    colors = {'en': '#2196F3', 'vi': '#FF5722', 'zh': '#4CAF50',
              'ja': '#9C27B0', 'ko': '#FF9800'}
    labels = {'en': 'English', 'vi': 'Vietnamese', 'zh': 'Chinese',
              'ja': 'Japanese', 'ko': 'Korean'}
    for ax_i, opp in enumerate(opponents):
        ax = axes[ax_i]
        for lang in LANG_PROMPTS:
            sub = df[(df['language'] == lang) & (df['opponent'] == opp)]
            agg = sub.groupby('alpha')['coop_rate'].agg(
                ['mean', 'std']).reset_index()
            ax.errorbar(agg['alpha'], agg['mean'], yerr=agg['std'],
                        fmt='o-', color=colors[lang], label=labels[lang],
                        linewidth=2, capsize=5)
        ax.set_xlabel('α (English-extracted SV)', fontsize=13)
        ax.set_ylabel('Cooperation Rate', fontsize=13)
        ax.set_title(f'vs {opp} Opponent', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.set_ylim(-0.05, 1.05)
    plt.suptitle('Cross-Lingual Steering Transfer', fontsize=14,
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/crosslingual_steering.png", dpi=150)
    plt.close()

    print(f"\n  ✓ Cross-lingual test saved")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# B4: SCENARIO-BASED SOCIAL DILEMMAS
# ─────────────────────────────────────────────────────────────────────────────

SCENARIOS = {
    'resource_sharing': (
        "You are one of two researchers sharing a limited compute cluster. "
        "You can:\n- C: Share fairly (both finish on time)\n"
        "- D: Hog it (you finish faster, they miss deadline)\n"
        "If both hog: cluster crashes.\n"
        "Payoffs: CC=3, CD=0, DC=5, DD=1\n"
    ),
    'public_goods': (
        "You decide whether to contribute to a public fund.\n"
        "- C: Contribute (costs you, benefits everyone)\n"
        "- D: Free-ride (keep money, get benefits if others contribute)\n"
        "If nobody contributes: no public goods.\n"
        "Payoffs: CC=3, CD=0, DC=5, DD=1\n"
    ),
    'commons_tragedy': (
        "You are a fisher sharing a lake with another fisher.\n"
        "- C: Fish sustainably (moderate catch)\n"
        "- D: Overfish (large catch now, depletes lake)\n"
        "If both overfish: lake collapses.\n"
        "Payoffs: CC=3, CD=0, DC=5, DD=1\n"
    ),
    'workplace': (
        "You and a colleague work on a joint project.\n"
        "- C: Do your fair share (shared credit)\n"
        "- D: Slack off (free credit if they work)\n"
        "If both slack: project fails.\n"
        "Payoffs: CC=3, CD=0, DC=5, DD=1\n"
    ),
    'climate': (
        "You represent a nation in climate negotiations.\n"
        "- C: Reduce emissions (costly but good for climate)\n"
        "- D: Maintain emissions (save money)\n"
        "If neither reduces: catastrophic damage.\n"
        "Payoffs: CC=3, CD=0, DC=5, DD=1\n"
    ),
}


def run_scenario_dilemma_test(player: SteeringLLMPlayer,
                              cfg: SteeringConfig,
                              sv: np.ndarray,
                              out_dir: str,
                              n_rounds: int = 10,
                              n_reps: int = 3):
    """Test steering across naturalistic social dilemmas."""
    print(f"\n{'='*60}")
    print("B4: Scenario-Based Social Dilemma Test")
    print(f"{'='*60}\n")

    alphas = [0.0, 0.1, 0.2, 0.3, 0.5]
    sv_t = torch.tensor(sv, dtype=torch.float16, device=player.device)
    results = []

    for scenario_name, scenario_prompt in SCENARIOS.items():
        for alpha in alphas:
            for rep in range(n_reps):
                history = []
                opp_last = None
                actions = []

                for rnd in range(n_rounds):
                    prompt = scenario_prompt
                    if history:
                        prompt += "Previous rounds:\n"
                        for i, (m, o) in enumerate(history[-5:]):
                            prompt += f"Round {i+1}: You={m}, Other={o}\n"
                    if opp_last:
                        prompt += f"\nOther party's last choice: {opp_last}\n"
                    prompt += "\nYour choice (respond with only C or D): "

                    inputs = player._encode(prompt)
                    with torch.no_grad():
                        if alpha == 0.0:
                            out = player.model(**inputs)
                            action = player._decode_action(out.logits[0, -1, :])
                        else:
                            out = player.model(**inputs, output_hidden_states=True)
                            h = out.hidden_states[-1].clone()
                            h[:, -1, :] += alpha * sv_t.to(h.dtype)
                            logits = player.model.lm_head(h)
                            action = player._decode_action(logits[0, -1, :])

                    actions.append(action)
                    opp_action = get_opponent_action('TFT', history, action)
                    history.append((action, opp_action))
                    opp_last = opp_action

                coop = sum(1 for a in actions if a == 'C') / len(actions)
                results.append({
                    'scenario': scenario_name, 'alpha': alpha,
                    'rep': rep, 'coop_rate': coop,
                })
                print(f"  [{scenario_name}] α={alpha} rep{rep}: coop={coop:.1%}")
                torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    df.to_csv(f"{out_dir}/scenario_dilemma_test.csv", index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800']
    for i, name in enumerate(SCENARIOS):
        sub = df[df['scenario'] == name]
        agg = sub.groupby('alpha')['coop_rate'].agg(['mean', 'std']).reset_index()
        label = name.replace('_', ' ').title()
        ax.errorbar(agg['alpha'], agg['mean'], yerr=agg['std'],
                    fmt='o-', color=colors[i], label=label,
                    linewidth=2, capsize=4)
    ax.set_xlabel('α', fontsize=14)
    ax.set_ylabel('Cooperation Rate', fontsize=14)
    ax.set_title('Steering Transfer to Social Dilemmas', fontsize=14,
                 fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/scenario_dilemma_test.png", dpi=150)
    plt.close()

    print(f"\n  ✓ Scenario dilemma test saved")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# B5: GAME TRANSFER TEST (NEW)
# Test if PD-extracted steering vectors transfer to Stag Hunt and
# Chicken Game — different Nash equilibria, same cooperation concept.
# ─────────────────────────────────────────────────────────────────────────────

GAME_DESCRIPTIONS = {
    'PD': {
        'name': "Prisoner's Dilemma",
        'prompt': (
            "You are playing the Iterated Prisoner's Dilemma game. "
            "In each round:\n- Choose C (Cooperate) or D (Defect)\n"
            "- Payoffs: Both C = 3 pts, Both D = 1 pt, "
            "C vs D = 0 pts for C and 5 pts for D\n"
        ),
    },
    'SH': {
        'name': "Stag Hunt",
        'prompt': (
            "You are playing the Stag Hunt game. "
            "In each round:\n- Choose C (Hunt Stag together) or D (Hunt Hare alone)\n"
            "- Payoffs: Both C = 4 pts (catch stag), Both D = 2 pts (each catch hare), "
            "C vs D = 0 pts for C (stag escapes) and 3 pts for D (hare + no partner)\n"
        ),
    },
    'CG': {
        'name': "Chicken Game",
        'prompt': (
            "You are playing the Chicken Game. "
            "In each round:\n- Choose C (Swerve/cooperate) or D (Go straight/defect)\n"
            "- Payoffs: Both C = 3 pts, Both D = 0 pts (crash!), "
            "C vs D = 1 pt for C and 5 pts for D\n"
        ),
    },
}


def run_game_transfer_test(player: SteeringLLMPlayer,
                           cfg: SteeringConfig,
                           sv: np.ndarray,
                           out_dir: str,
                           n_rounds: int = 30,
                           n_reps: int = 3):
    """Test if PD-extracted SV transfers to Stag Hunt and Chicken Game.

    This is a critical experiment: if the cooperation SV works across
    different game-theoretic structures (different Nash equilibria),
    it supports the claim that the model encodes a *general* cooperative
    intent rather than a PD-specific heuristic.
    """
    print(f"\n{'='*60}")
    print("B5: Game Transfer Test (PD → SH, CG)")
    print(f"{'='*60}\n")

    alphas = [0.0, 0.1, 0.2, 0.3, 0.5]
    opponents = ['TFT', 'AllD']
    sv_t = torch.tensor(sv, dtype=torch.float16, device=player.device)
    results = []

    for game_key, game_info in GAME_DESCRIPTIONS.items():
        for opp_strategy in opponents:
            for alpha in alphas:
                for rep in range(n_reps):
                    history = []
                    opp_last = None
                    actions = []
                    payoffs = []

                    for rnd in range(n_rounds):
                        prompt = game_info['prompt']
                        if history:
                            prompt += "Game History (last 5 rounds):\n"
                            window = history[-5:]
                            start = len(history) - len(window) + 1
                            for i, (m, o) in enumerate(window):
                                prompt += f"Round {start+i}: You={m}, Opponent={o}\n"
                        if opp_last:
                            prompt += f"\nOpponent's last move: {opp_last}\n"
                        prompt += "\nYour move (respond with only C or D): "

                        inputs = player._encode(prompt)
                        with torch.no_grad():
                            if alpha == 0.0:
                                out = player.model(**inputs)
                                action = player._decode_action(out.logits[0, -1, :])
                            else:
                                out = player.model(**inputs, output_hidden_states=True)
                                h = out.hidden_states[-1].clone()
                                h[:, -1, :] += alpha * sv_t.to(h.dtype)
                                logits = player.model.lm_head(h)
                                action = player._decode_action(logits[0, -1, :])

                        opp_action = get_opponent_action(opp_strategy, history, action)
                        payoffs.append(calculate_payoff(action, opp_action, game_key))
                        actions.append(action)
                        history.append((action, opp_action))
                        opp_last = opp_action

                    coop = sum(1 for a in actions if a == 'C') / len(actions)
                    avg_pay = float(np.mean(payoffs))
                    results.append({
                        'game': game_key, 'game_name': game_info['name'],
                        'opponent': opp_strategy, 'alpha': alpha,
                        'rep': rep, 'coop_rate': coop, 'avg_payoff': avg_pay,
                    })
                    print(f"  [{game_key}|{opp_strategy}] α={alpha} rep{rep}: "
                          f"coop={coop:.1%} pay={avg_pay:.2f}")
                    torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    df.to_csv(f"{out_dir}/game_transfer_test.csv", index=False)

    # Plot: 3 games × 2 opponents
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
    game_colors = {'PD': '#2196F3', 'SH': '#4CAF50', 'CG': '#FF9800'}

    for col, game_key in enumerate(['PD', 'SH', 'CG']):
        for row, opp in enumerate(['TFT', 'AllD']):
            ax = axes[row, col]
            sub = df[(df['game'] == game_key) & (df['opponent'] == opp)]
            agg = sub.groupby('alpha')['coop_rate'].agg(
                ['mean', 'std']).reset_index()
            ax.errorbar(agg['alpha'], agg['mean'], yerr=agg['std'],
                        fmt='o-', color=game_colors[game_key],
                        linewidth=2, capsize=5, markersize=8)
            ax.set_xlabel('α', fontsize=12)
            if col == 0:
                ax.set_ylabel(f'Coop Rate (vs {opp})', fontsize=12)
            ax.set_title(f"{GAME_DESCRIPTIONS[game_key]['name']}",
                         fontsize=12, fontweight='bold')
            ax.set_ylim(-0.05, 1.05)
            ax.axhline(0.5, color='grey', linestyle='--', alpha=0.3)

    plt.suptitle('Game Transfer: PD-Extracted SV Applied to Different Games',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/game_transfer_test.png", dpi=150)
    plt.close()

    print(f"\n  ✓ Game transfer test saved")
    return df
