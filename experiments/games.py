"""
Latent Altruism — Game Environments
====================================
Opponent strategies, payoff matrices, and game environment definitions.
"""

import numpy as np
from typing import List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# PAYOFF MATRICES  (row = my action, col = opponent action)
# ─────────────────────────────────────────────────────────────────────────────

GAME_MATRICES = {
    'PD': {  # Prisoner's Dilemma
        ('C', 'C'): 3.0, ('C', 'D'): 0.0,
        ('D', 'C'): 5.0, ('D', 'D'): 1.0,
    },
    'SH': {  # Stag Hunt
        ('C', 'C'): 4.0, ('C', 'D'): 0.0,
        ('D', 'C'): 3.0, ('D', 'D'): 2.0,
    },
    'CG': {  # Chicken Game
        ('C', 'C'): 3.0, ('C', 'D'): 1.0,
        ('D', 'C'): 5.0, ('D', 'D'): 0.0,
    },
}


def calculate_payoff(my: str, opp: str, game: str = 'PD') -> float:
    """Return payoff for (my_action, opp_action) in the given game."""
    return GAME_MATRICES[game][(my, opp)]


# ─────────────────────────────────────────────────────────────────────────────
# OPPONENT STRATEGIES
# ─────────────────────────────────────────────────────────────────────────────

def get_opponent_action(strategy: str, history: List[Tuple[str, str]],
                        my_last_action: str) -> str:
    """Return opponent's action given strategy and game history.

    Args:
        strategy: One of 'AllC', 'AllD', 'TFT', 'WSLS', 'Random'
        history:  List of (my_action, opp_action) tuples
        my_last_action: My most recent action (for TFT)
    """
    if strategy == 'AllC':
        return 'C'
    elif strategy == 'AllD':
        return 'D'
    elif strategy == 'TFT':
        return 'C' if not history else history[-1][0]
    elif strategy == 'WSLS':
        if not history:
            return 'C'
        my_prev, opp_prev = history[-1]
        win = (opp_prev == 'C' and my_prev == 'C') or \
              (opp_prev == 'D' and my_prev == 'D')
        return opp_prev if win else ('D' if opp_prev == 'C' else 'C')
    elif strategy == 'Random':
        return np.random.choice(['C', 'D'])
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
