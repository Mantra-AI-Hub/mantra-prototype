"""Self-evolving recommender system with retraining and strategy switching."""

from __future__ import annotations

import logging
from collections import deque
from typing import Callable, Deque, Dict, List, Sequence, Tuple


StrategyFn = Callable[[Sequence[Tuple[str, float]], Dict[str, object] | None], List[Tuple[str, float]]]


class SelfEvolvingRecommender:
    def __init__(
        self,
        strategies: Dict[str, StrategyFn] | None = None,
        retrain_threshold: float = 0.55,
        window_size: int = 20,
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.strategies = strategies or {"score_desc": self._score_desc_strategy}
        self.current_strategy = next(iter(self.strategies))
        self.retrain_threshold = float(retrain_threshold)
        self.performance_window: Deque[float] = deque(maxlen=max(3, int(window_size)))
        self.metrics: Dict[str, float | int | str] = {
            "rank_calls": 0,
            "retrain_calls": 0,
            "switch_calls": 0,
            "avg_performance": 0.0,
            "current_strategy": self.current_strategy,
        }

    def _score_desc_strategy(
        self, candidates: Sequence[Tuple[str, float]], context: Dict[str, object] | None = None
    ) -> List[Tuple[str, float]]:
        return sorted([(str(t), float(s)) for t, s in candidates], key=lambda x: x[1], reverse=True)

    def register_strategy(self, name: str, strategy_fn: StrategyFn) -> None:
        self.strategies[str(name)] = strategy_fn
        self.logger.info("Registered strategy %s", name)

    def rank(
        self, candidates: Sequence[Tuple[str, float]], context: Dict[str, object] | None = None
    ) -> List[Tuple[str, float]]:
        self.metrics["rank_calls"] = int(self.metrics["rank_calls"]) + 1
        strategy = self.strategies[self.current_strategy]
        ranked = strategy(candidates, context)
        self.logger.debug("Ranked %d candidates with %s", len(ranked), self.current_strategy)
        return ranked

    def monitor_model_performance(self, score: float) -> Dict[str, float | int | str]:
        bounded = max(0.0, min(1.0, float(score)))
        self.performance_window.append(bounded)
        avg_perf = sum(self.performance_window) / max(1, len(self.performance_window))
        self.metrics["avg_performance"] = float(avg_perf)
        self.logger.info("Performance updated: score=%.4f avg=%.4f", bounded, avg_perf)
        return self.metrics_snapshot()

    def should_retrain(self) -> bool:
        avg_perf = float(self.metrics["avg_performance"])
        should = len(self.performance_window) >= 3 and avg_perf < self.retrain_threshold
        return bool(should)

    def auto_retrain_models(self, retrain_fn: Callable[[], bool] | None = None) -> bool:
        if not self.should_retrain():
            return False
        self.metrics["retrain_calls"] = int(self.metrics["retrain_calls"]) + 1
        result = bool(retrain_fn()) if retrain_fn is not None else True
        self.logger.warning("Auto retrain triggered, success=%s", result)
        return result

    def auto_switch_ranking_strategies(self, recommended_strategy: str | None = None) -> str:
        target = str(recommended_strategy or self.current_strategy)
        if target not in self.strategies:
            self.logger.warning("Unknown strategy %s, keeping %s", target, self.current_strategy)
            return self.current_strategy
        if target != self.current_strategy:
            self.current_strategy = target
            self.metrics["switch_calls"] = int(self.metrics["switch_calls"]) + 1
            self.metrics["current_strategy"] = self.current_strategy
            self.logger.info("Switched ranking strategy to %s", target)
        return self.current_strategy

    def metrics_snapshot(self) -> Dict[str, float | int | str]:
        return dict(self.metrics)
