"""
core.run_io
-----------
Single source of truth for run identity and artifact filenames.

A :class:`RunIO` instance binds together:

* a problem class name (the canonical first level of the
  ``results/<ProblemClassName>/`` tree),
* a ``tag`` describing the kind of run (``"JFB"``, ``"FullAD"``, ...),
* a stable ``run_id`` (defaults to a wall-clock timestamp at construction),
* and the canonical filename templates for every artifact a training run
  produces.

The point is to keep the example runners free of any path or filename
reasoning. They build a problem, build a policy, hand both to the
trainer; the trainer constructs / receives a :class:`RunIO`; the
:class:`RunIO` decides where every byte lands.

Layout produced by a single training run::

    results/<ProblemClassName>/
    ├── training/
    │   ├── best_policy_<stem>.pth
    │   ├── history_<stem>.csv
    │   ├── loss_curve_<stem>.png
    │   └── training-plots/
    │       └── rollout_<stem>_NNNN.png
    └── rollouts/
        ├── policy_rollout_<stem>.png
        └── trajectory_<stem>.pth

where ``<stem> = f"{tag}_{run_id}"``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime

from core.paths import results_dir


def _default_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass
class RunIO:
    """Path/filename policy for a single training run."""

    problem_cls_name: str
    tag: str = "JFB"
    run_id: str = field(default_factory=_default_run_id)

    @property
    def stem(self) -> str:
        """The shared filename prefix, e.g. ``"JFB_20260425_154755"``."""
        return f"{self.tag}_{self.run_id}"

    # ------------------------------------------------------------------
    # Directories (each created on demand by ``results_dir``).
    # ------------------------------------------------------------------
    @property
    def train_dir(self) -> str:
        return results_dir(self.problem_cls_name, "training")

    @property
    def plots_dir(self) -> str:
        return results_dir(self.problem_cls_name, "training", "training-plots")

    @property
    def rollout_dir(self) -> str:
        return results_dir(self.problem_cls_name, "rollouts")

    @property
    def benchmark_dir(self) -> str:
        return results_dir(self.problem_cls_name, "benchmark")

    @property
    def reference_dir(self) -> str:
        return results_dir(self.problem_cls_name, "reference")

    # ------------------------------------------------------------------
    # Per-artifact filenames.
    # ------------------------------------------------------------------
    def policy_path(self) -> str:
        return os.path.join(self.train_dir, f"best_policy_{self.stem}.pth")

    def history_path(self) -> str:
        return os.path.join(self.train_dir, f"history_{self.stem}.csv")

    def loss_curve_path(self) -> str:
        return os.path.join(self.train_dir, f"loss_curve_{self.stem}.png")

    def training_plot_path(self, epoch: int) -> str:
        return os.path.join(self.plots_dir, f"rollout_{self.stem}_{epoch:04d}.png")

    def rollout_path(self) -> str:
        return os.path.join(self.rollout_dir, f"policy_rollout_{self.stem}.png")

    def trajectory_path(self) -> str:
        return os.path.join(self.rollout_dir, f"trajectory_{self.stem}.pth")
