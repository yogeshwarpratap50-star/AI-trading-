"""
Hyperparameter Optimizer — Phase 11C.

Supports Grid Search, Random Search, and Optuna (if installed).
Always falls back to random search if optuna is unavailable.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score


def _try_optuna():
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        return optuna
    except ImportError:
        return None


@dataclass
class OptimizationResult:
    best_params: dict[str, Any]
    best_score: float
    method: str
    n_trials: int
    all_results: list[dict[str, Any]] = field(default_factory=list)


class HyperparameterOptimizer:
    """
    Finds best hyperparameters for an sklearn-compatible estimator.

    Usage (Optuna)
    --------------
    opt = HyperparameterOptimizer(estimator_fn, param_space, method="optuna")
    result = opt.optimize(X, y, n_trials=50)

    estimator_fn(**params) must return a fitted/unfitted sklearn estimator.
    param_space: dict of param_name → (low, high) for numeric,
                 or list of values for categorical.
    """

    def __init__(
        self,
        estimator_fn: Callable[..., Any],
        param_space: dict[str, Any],
        method: str = "optuna",
        scoring: str = "accuracy",
        cv: int = 3,
    ) -> None:
        self.estimator_fn = estimator_fn
        self.param_space = param_space
        self.method = method
        self.scoring = scoring
        self.cv = cv
        self.logger = logging.getLogger(__name__)

    def optimize(self, X: pd.DataFrame, y: pd.Series, n_trials: int = 50) -> OptimizationResult:
        if self.method == "grid":
            return self._grid_search(X, y)
        if self.method == "optuna":
            optuna = _try_optuna()
            if optuna:
                return self._optuna_search(X, y, optuna, n_trials)
            self.logger.warning("Optuna not installed — falling back to random search")
        return self._random_search(X, y, n_trials)

    # ------------------------------------------------------------------

    def _grid_search(self, X: pd.DataFrame, y: pd.Series) -> OptimizationResult:
        from itertools import product as itertools_product
        keys = list(self.param_space.keys())
        values = [v if isinstance(v, list) else [v] for v in self.param_space.values()]
        best_score = -np.inf
        best_params: dict = {}
        all_results = []
        for combo in itertools_product(*values):
            params = dict(zip(keys, combo))
            score = self._score(params, X, y)
            all_results.append({**params, "score": score})
            if score > best_score:
                best_score = score
                best_params = params
        return OptimizationResult(best_params, best_score, "grid", len(all_results), all_results)

    def _random_search(self, X: pd.DataFrame, y: pd.Series, n_trials: int) -> OptimizationResult:
        best_score = -np.inf
        best_params: dict = {}
        all_results = []
        for _ in range(n_trials):
            params = self._sample_params()
            score = self._score(params, X, y)
            all_results.append({**params, "score": score})
            if score > best_score:
                best_score = score
                best_params = params
        return OptimizationResult(best_params, best_score, "random", n_trials, all_results)

    def _optuna_search(self, X: pd.DataFrame, y: pd.Series, optuna: Any, n_trials: int) -> OptimizationResult:
        all_results: list[dict] = []

        def objective(trial):
            params = {}
            for name, space in self.param_space.items():
                if isinstance(space, list):
                    params[name] = trial.suggest_categorical(name, space)
                elif isinstance(space, tuple) and len(space) == 2:
                    lo, hi = space
                    if isinstance(lo, int) and isinstance(hi, int):
                        params[name] = trial.suggest_int(name, lo, hi)
                    else:
                        params[name] = trial.suggest_float(name, lo, hi)
                else:
                    params[name] = space
            score = self._score(params, X, y)
            all_results.append({**params, "score": score})
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        return OptimizationResult(
            study.best_params, study.best_value, "optuna", n_trials, all_results
        )

    def _score(self, params: dict, X: pd.DataFrame, y: pd.Series) -> float:
        try:
            estimator = self.estimator_fn(**params)
            scores = cross_val_score(estimator, X, y, cv=self.cv, scoring=self.scoring)
            return float(np.mean(scores))
        except Exception as exc:
            self.logger.debug("Trial failed: %s", exc)
            return -np.inf

    def _sample_params(self) -> dict:
        params = {}
        for name, space in self.param_space.items():
            if isinstance(space, list):
                params[name] = random.choice(space)
            elif isinstance(space, tuple) and len(space) == 2:
                lo, hi = space
                if isinstance(lo, int) and isinstance(hi, int):
                    params[name] = random.randint(lo, hi)
                else:
                    params[name] = random.uniform(lo, hi)
            else:
                params[name] = space
        return params
