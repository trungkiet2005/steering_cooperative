# Latent Altruism — Experiment Package
from .config import SteeringConfig, MODEL_REGISTRY, get_model_config
from .games import get_opponent_action, calculate_payoff, GAME_MATRICES
from .model import SteeringLLMPlayer
from .runner import run_all_experiments, run_fdi_sweep
from .benchmarks import (
    compute_perplexity_standard,
    run_semantic_invariance_test,
    run_crosslingual_test,
    run_scenario_dilemma_test,
    run_game_transfer_test,
)
