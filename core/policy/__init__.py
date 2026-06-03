"""Non-RL policies: greedy heuristic and brute-force optimal."""
from .heuristic import greedy_allocate
from .optimizer import optimal_allocate

__all__ = ["greedy_allocate", "optimal_allocate"]
