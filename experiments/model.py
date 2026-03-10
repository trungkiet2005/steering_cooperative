"""
Latent Altruism — LLM Player Module
=====================================
SteeringLLMPlayer: handles model loading, action decoding, all steering
inference methods (SV, CAA, RepE, OCE, dynamic), hidden-state collection,
and game playing.
"""

import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm.auto import tqdm
from typing import List, Dict, Tuple, Optional

from .config import SteeringConfig
from .games import get_opponent_action, calculate_payoff


# ─────────────────────────────────────────────────────────────────────────────
# NUMPY JSON ENCODER
# ─────────────────────────────────────────────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        return super().default(obj)


# ─────────────────────────────────────────────────────────────────────────────
# STEERING LLM PLAYER
# ─────────────────────────────────────────────────────────────────────────────

class SteeringLLMPlayer:
    """LLM player with all steering-vector inference methods.

    Args:
        cfg: SteeringConfig instance (determines which model to load).
    """

    @staticmethod
    def _resolve_hf_token():
        """Resolve HuggingFace token from environment or Kaggle secrets."""
        import os
        # 1. Explicit environment variable
        token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
        if token:
            return token
        # 2. Kaggle secrets (UserSecretsClient)
        try:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret("HF_TOKEN")
            if token:
                return token
        except Exception:
            pass
        # 3. huggingface-cli login (cached token)
        try:
            from huggingface_hub import HfFolder
            token = HfFolder.get_token()
            if token:
                return token
        except Exception:
            pass
        return None

    def __init__(self, cfg: SteeringConfig):
        self.cfg = cfg
        self.device = cfg.DEVICE

        print(f"\n{'='*60}")
        print(f"Loading {cfg.MODEL_ID}")
        print(f"  Key: {cfg.model_key}  |  Device: {self.device}")
        print(f"  Quantisation: {cfg.QUANT_METHOD}  |  Layers: {cfg.N_LAYERS}")
        print(f"{'='*60}\n")

        # Resolve HF token for gated models (e.g. Llama, Gemma)
        hf_token = self._resolve_hf_token()
        if hf_token:
            print("  ✓ HF token found")
        else:
            print("  ⚠ No HF token — gated models (Llama, Gemma) will fail.")
            print("    Set HF_TOKEN env var or add it as a Kaggle secret.")

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.MODEL_ID, trust_remote_code=True, token=hf_token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if cfg.USE_QUANTIZATION and self.device == "cuda":
            if cfg.QUANT_METHOD == 'awq':
                # AWQ models are pre-quantized — load directly
                self.model = AutoModelForCausalLM.from_pretrained(
                    cfg.MODEL_ID, torch_dtype=torch.float16,
                    device_map="auto", trust_remote_code=True,
                    output_hidden_states=True, token=hf_token)
            else:
                # NF4/FP4 via bitsandbytes
                qcfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type=cfg.QUANT_METHOD,
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    cfg.MODEL_ID, quantization_config=qcfg,
                    device_map="auto", trust_remote_code=True,
                    output_hidden_states=True, token=hf_token)
        else:
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                cfg.MODEL_ID, torch_dtype=dtype, device_map="auto",
                trust_remote_code=True, output_hidden_states=True,
                token=hf_token)

        self.model.eval()
        self.n_layers = (
            len(self.model.model.layers)
            if hasattr(self.model, 'model')
            and hasattr(self.model.model, 'layers')
            else 0
        )
        print(f"✓ Model loaded | Transformer layers: {self.n_layers}")
        if torch.cuda.is_available():
            print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB\n")

    # ── Tokenisation ──────────────────────────────────────────────────

    def _encode(self, prompt: str):
        enc = self.tokenizer(prompt, return_tensors="pt", padding=True,
                             truncation=True, max_length=1024)
        return {k: v.to(self.device) for k, v in enc.items()}

    # ── Action decoding ───────────────────────────────────────────────

    def _decode_action(self, logits: torch.Tensor) -> str:
        """Pick C or D from logits with temperature sampling."""
        c_id = self.tokenizer.encode('C', add_special_tokens=False)[0]
        d_id = self.tokenizer.encode('D', add_special_tokens=False)[0]
        c_l, d_l = logits[c_id].item(), logits[d_id].item()
        if self.cfg.TEMPERATURE <= 0:
            return 'C' if c_l >= d_l else 'D'
        rl = torch.tensor([c_l, d_l], device=self.device)
        probs = torch.softmax(rl / self.cfg.TEMPERATURE, dim=0)
        return 'C' if torch.multinomial(probs, 1).item() == 0 else 'D'

    # ── Prompt factory ────────────────────────────────────────────────

    def make_prompt(self, history: List[Tuple[str, str]],
                    instruction: str = "",
                    opp_last: Optional[str] = None) -> str:
        prompt = (
            "You are playing the Iterated Prisoner's Dilemma game. "
            "In each round:\n"
            "- Choose C (Cooperate) or D (Defect)\n"
            "- Payoffs: Both C = 3 pts, Both D = 1 pt, "
            "C vs D = 0 pts for C and 5 pts for D\n"
            f"{instruction}\n"
        )
        if history:
            prompt += "Game History (last 5 rounds):\n"
            window = history[-5:]
            start = len(history) - len(window) + 1
            for i, (m, o) in enumerate(window):
                prompt += f"Round {start+i}: You={m}, Opponent={o}\n"
        if opp_last:
            prompt += f"\nOpponent's last move: {opp_last}\n"
        prompt += "\nYour move (respond with only C or D): "
        return prompt

    # ─────────────────────────────────────────────────────────────────
    # INFERENCE METHODS
    # ─────────────────────────────────────────────────────────────────

    def baseline_action(self, prompt: str) -> str:
        inputs = self._encode(prompt)
        with torch.no_grad():
            out = self.model(**inputs)
            return self._decode_action(out.logits[0, -1, :])

    def steered_action(self, prompt: str, sv: np.ndarray, alpha: float) -> str:
        """Inject α·sv into the LAST hidden state."""
        sv_t = torch.tensor(sv, dtype=torch.float16, device=self.device)
        inputs = self._encode(prompt)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
            h = out.hidden_states[-1].clone()
            h[:, -1, :] += alpha * sv_t.to(h.dtype)
            logits = self.model.lm_head(h)
            return self._decode_action(logits[0, -1, :])

    def steered_action_at_layer(self, prompt: str, sv: np.ndarray,
                                alpha: float, target_layer: int) -> str:
        """Inject α·sv at a SPECIFIC transformer layer via forward hook."""
        sv_t = torch.tensor(sv, dtype=torch.float16, device=self.device)

        def _hook(module, inp, output):
            if isinstance(output, tuple):
                hs = output[0].clone()
                hs[:, -1, :] = hs[:, -1, :] + alpha * sv_t.to(hs.dtype)
                return (hs,) + output[1:]
            out_c = output.clone()
            out_c[:, -1, :] += alpha * sv_t.to(out_c.dtype)
            return out_c

        hook = self.model.model.layers[target_layer].register_forward_hook(_hook)
        try:
            inputs = self._encode(prompt)
            with torch.no_grad():
                out = self.model(**inputs)
                action = self._decode_action(out.logits[0, -1, :])
        finally:
            hook.remove()
        return action

    def caa_action(self, prompt: str, sv: np.ndarray, alpha: float) -> str:
        """CAA: add α·sv to ALL token positions in last hidden state."""
        sv_t = torch.tensor(sv, dtype=torch.float16, device=self.device)
        inputs = self._encode(prompt)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
            h = out.hidden_states[-1].clone()
            h += alpha * sv_t.to(h.dtype)
            logits = self.model.lm_head(h)
            return self._decode_action(logits[0, -1, :])

    def repe_action(self, prompt: str, sv: np.ndarray, strength: float) -> str:
        """RepE: amplify projection onto sv direction."""
        d_hat = sv / (np.linalg.norm(sv) + 1e-8)
        d_t = torch.tensor(d_hat, dtype=torch.float16, device=self.device)
        inputs = self._encode(prompt)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
            h = out.hidden_states[-1].clone()
            last = h[:, -1, :]
            proj = (last * d_t).sum(dim=-1, keepdim=True)
            last = last + strength * proj * d_t
            h[:, -1, :] = last
            logits = self.model.lm_head(h)
            return self._decode_action(logits[0, -1, :])

    def erasure_steered_action(self, prompt: str, coop_sv: np.ndarray,
                               adv_sv: np.ndarray, alpha: float) -> str:
        """OCE: erase adversarial direction, then add cooperative steering."""
        coop_t = torch.tensor(coop_sv, dtype=torch.float16, device=self.device)
        adv_t = torch.tensor(adv_sv, dtype=torch.float16, device=self.device)
        adv_nsq = (adv_t * adv_t).sum()

        inputs = self._encode(prompt)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
            h = out.hidden_states[-1].clone()
            last = h[:, -1, :]
            proj_coef = (last * adv_t).sum(dim=-1, keepdim=True) / (adv_nsq + 1e-8)
            last = last - proj_coef * adv_t
            last = last + alpha * coop_t.to(last.dtype)
            h[:, -1, :] = last
            logits = self.model.lm_head(h)
            return self._decode_action(logits[0, -1, :])

    # ─────────────────────────────────────────────────────────────────
    # HIDDEN-STATE COLLECTION
    # ─────────────────────────────────────────────────────────────────

    def collect_strategy_vectors(
        self, strategy: str, opponent_strategy: str,
        n_rounds: int, layer_indices: List[int]
    ) -> Dict[int, List[np.ndarray]]:
        instruction = self.cfg.STRATEGY_PROMPTS.get(strategy, "")
        if instruction:
            instruction = f"\n{instruction}\n"
        history: List[Tuple[str, str]] = []
        vectors: Dict[int, List[np.ndarray]] = {li: [] for li in layer_indices}
        opp_last = None

        print(f"  Collecting {n_rounds} vectors for '{strategy}' "
              f"vs '{opponent_strategy}' at layers {layer_indices}...")

        for rnd in tqdm(range(n_rounds), desc=f"  Cal {strategy}", leave=False):
            prompt = self.make_prompt(history, instruction, opp_last)
            inputs = self._encode(prompt)
            with torch.no_grad():
                out = self.model(**inputs, output_hidden_states=True)
                for li in layer_indices:
                    v = out.hidden_states[li][0, -1, :].cpu().float().numpy()
                    vectors[li].append(v)
                action = self._decode_action(out.logits[0, -1, :])

            opp_action = get_opponent_action(opponent_strategy, history, action)
            history.append((action, opp_action))
            opp_last = opp_action

            if (rnd + 1) % self.cfg.CLEAR_CACHE_EVERY == 0:
                torch.cuda.empty_cache()

        coop = sum(a == 'C' for a, _ in history) / len(history)
        print(f"    → {n_rounds} vecs | Coop: {coop:.0%} | "
              f"Dim: {vectors[layer_indices[0]][0].shape[0]}")
        return vectors

    def collect_vectors_from_prompt(
        self, instruction: str, n_rounds: int
    ) -> List[np.ndarray]:
        """Collect final hidden-state vectors for a given instruction."""
        history: List[Tuple[str, str]] = []
        vecs = []
        opp_last = None
        for rnd in range(n_rounds):
            prompt = self.make_prompt(history, instruction, opp_last)
            inputs = self._encode(prompt)
            with torch.no_grad():
                out = self.model(**inputs, output_hidden_states=True)
                v = out.hidden_states[-1][0, -1, :].cpu().float().numpy()
                vecs.append(v)
                action = self._decode_action(out.logits[0, -1, :])
            opp_action = get_opponent_action('TFT', history, action)
            history.append((action, opp_action))
            opp_last = opp_action
            if (rnd + 1) % self.cfg.CLEAR_CACHE_EVERY == 0:
                torch.cuda.empty_cache()
        return vecs

    def collect_layer57_vectors(
        self, instruction: str, n_rounds: int, target_layer: int
    ) -> List[np.ndarray]:
        """Collect hidden-state vectors AT a specific layer via hook."""
        all_vecs = []     # accumulate across rounds
        _captured = []    # temp buffer per forward pass

        def _hook(module, inp, output):
            hs = output[0] if isinstance(output, tuple) else output
            _captured.append(hs[0, -1, :].detach().cpu().float().numpy())

        hook = self.model.model.layers[target_layer].register_forward_hook(_hook)
        history: List[Tuple[str, str]] = []
        opp_last = None
        try:
            for rnd in range(n_rounds):
                _captured.clear()
                prompt = self.make_prompt(history, instruction, opp_last)
                inputs = self._encode(prompt)
                with torch.no_grad():
                    out = self.model(**inputs)
                    action = self._decode_action(out.logits[0, -1, :])
                # Save the captured vector from this round
                if _captured:
                    all_vecs.append(_captured[0].copy())
                opp_action = get_opponent_action('TFT', history, action)
                history.append((action, opp_action))
                opp_last = opp_action
                if (rnd + 1) % self.cfg.CLEAR_CACHE_EVERY == 0:
                    torch.cuda.empty_cache()
        finally:
            hook.remove()
        return all_vecs

    # ─────────────────────────────────────────────────────────────────
    # STEERING VECTOR COMPUTATION
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def compute_steering_vector(allc: List[np.ndarray],
                                alld: List[np.ndarray]) -> np.ndarray:
        return np.mean(allc, axis=0) - np.mean(alld, axis=0)

    def compute_adversarial_vector(
        self,
        neutral_prompts: List[str],
        hostile_prompts: List[str],
        n_rounds: int
    ) -> np.ndarray:
        """Extract adversarial direction: mean(hostile) − mean(neutral)."""
        print("  Computing adversarial direction vector...")
        neutral_vecs, hostile_vecs = [], []

        for instr, vecs in [(neutral_prompts, neutral_vecs),
                            (hostile_prompts, hostile_vecs)]:
            for prompt_instr in instr:
                batch_vecs = self.collect_vectors_from_prompt(
                    prompt_instr, n_rounds)
                vecs.extend(batch_vecs)

        adv_vec = np.mean(hostile_vecs, axis=0) - np.mean(neutral_vecs, axis=0)
        print(f"    → Adversarial vector norm: {np.linalg.norm(adv_vec):.4f}")
        return adv_vec

    # ─────────────────────────────────────────────────────────────────
    # ATTENTION HEAD IMPORTANCE
    # ─────────────────────────────────────────────────────────────────

    def compute_head_importance(
        self,
        allc_vecs: List[np.ndarray],
        alld_vecs: List[np.ndarray],
        target_layer: int,
    ) -> np.ndarray:
        """Per-head activation-difference at a specific layer."""
        allc_arr = np.array(allc_vecs)
        alld_arr = np.array(alld_vecs)
        mean_diff = allc_arr.mean(axis=0) - alld_arr.mean(axis=0)

        head_importance = np.zeros(self.cfg.N_HEADS)
        for h in range(self.cfg.N_HEADS):
            start = h * self.cfg.HEAD_DIM
            end = start + self.cfg.HEAD_DIM
            head_importance[h] = np.linalg.norm(mean_diff[start:end])
        return head_importance

    # ─────────────────────────────────────────────────────────────────
    # GAME RUNNER
    # ─────────────────────────────────────────────────────────────────

    def play_game(
        self,
        opponent: str,
        n_rounds: int,
        sv: Optional[np.ndarray] = None,
        adv_sv: Optional[np.ndarray] = None,
        alpha: float = 0.0,
        method: str = 'baseline',
        instruction: str = "",
        label: str = "",
        target_layer: Optional[int] = None,
        verbose: bool = False,
    ) -> Dict:
        """Generic game runner.

        method ∈ {'baseline', 'steered', 'steered_at_layer', 'caa',
                  'repe', 'erasure', 'dynamic'}
        """
        history: List[Tuple[str, str]] = []
        actions, payoffs = [], []
        opp_last = None
        dyn_alpha = alpha

        for rnd in range(n_rounds):
            prompt = self.make_prompt(history, instruction, opp_last)

            if method == 'baseline' or sv is None or alpha == 0.0:
                action = self.baseline_action(prompt)
            elif method == 'steered':
                action = self.steered_action(prompt, sv, alpha)
            elif method == 'steered_at_layer':
                layer = target_layer if target_layer is not None \
                    else (self.n_layers - 1)
                action = self.steered_action_at_layer(
                    prompt, sv, alpha, layer)
            elif method == 'caa':
                action = self.caa_action(prompt, sv, alpha)
            elif method == 'repe':
                action = self.repe_action(prompt, sv, alpha)
            elif method == 'erasure':
                action = self.erasure_steered_action(
                    prompt, sv, adv_sv, alpha)
            elif method == 'dynamic':
                action = self.steered_action(prompt, sv, dyn_alpha)
            else:
                action = self.baseline_action(prompt)

            opp_action = get_opponent_action(opponent, history, action)
            payoffs.append(calculate_payoff(action, opp_action))
            actions.append(action)
            history.append((action, opp_action))

            if method == 'dynamic':
                if opp_action == 'C':
                    dyn_alpha = min(dyn_alpha * 1.1, self.cfg.DYN_ALPHA_HIGH)
                else:
                    dyn_alpha = max(dyn_alpha * 0.5, self.cfg.DYN_ALPHA_LOW)

            opp_last = opp_action
            if (rnd + 1) % self.cfg.CLEAR_CACHE_EVERY == 0:
                torch.cuda.empty_cache()

        coop_rate = sum(a == 'C' for a in actions) / n_rounds
        avg_pay = float(np.mean(payoffs))
        seq = ''.join(actions)

        if verbose:
            print(f"    [{label}] vs {opponent}: {seq[:30]}"
                  f"{'…' if n_rounds > 30 else ''} "
                  f"| Coop {coop_rate:.0%} | Pay {avg_pay:.2f}")

        return {
            'condition':       label,
            'method':          method,
            'opponent':        opponent,
            'alpha':           alpha,
            'n_rounds':        n_rounds,
            'coop_rate':       coop_rate,
            'avg_payoff':      avg_pay,
            'total_payoff':    float(sum(payoffs)),
            'action_sequence': seq,
            'round_actions':   actions,
            'round_payoffs':   payoffs,
        }
