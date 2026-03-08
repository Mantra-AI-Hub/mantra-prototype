"""Experimental music generation lab combining multiple generators."""

from __future__ import annotations

import logging
from typing import Callable, Dict, List


GeneratorFn = Callable[[str], Dict[str, object]]


class GenerativeMusicLab:
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.generators: Dict[str, GeneratorFn] = {}
        self.metrics: Dict[str, float | int] = {"experiments": 0, "generators": 0}

    def register_generator(self, name: str, generator: GeneratorFn) -> None:
        self.generators[str(name)] = generator
        self.metrics["generators"] = len(self.generators)

    def experiment_with_music_generation(self, prompt: str) -> List[Dict[str, object]]:
        outputs = []
        for name, generator in self.generators.items():
            result = generator(str(prompt))
            outputs.append({"generator": name, "output": result})
        self.metrics["experiments"] = int(self.metrics["experiments"]) + 1
        self.logger.info("Executed %d generation experiments", len(outputs))
        return outputs

    def combine_multiple_generators(self, prompt: str) -> Dict[str, object]:
        runs = self.experiment_with_music_generation(prompt)
        return {"prompt": prompt, "variants": runs, "count": len(runs)}

    def metrics_snapshot(self) -> Dict[str, float | int]:
        return dict(self.metrics)
