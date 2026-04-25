"""
core.paths
----------
Single source of truth for filesystem paths under
``jfb-for-implicit-oc/results/``.

Every writer in the codebase (trainer, problem classes, examples,
benchmark/plotting scripts) goes through :func:`results_dir` so the
output location is fully decoupled from the current working directory.
The base directory is anchored to this file's location, which means
running scripts from anywhere on disk lands artifacts in the same
canonical place.

Layout produced by :func:`results_dir`::

    results/<ProblemClassName>/
    ├── training/                 # produced during a training run
    │   └── training-plots/       # mid-training rollout snapshots
    ├── rollouts/                 # post-training inference of the trained policy
    ├── reference/                # analytical / numerical reference (no ML model)
    └── benchmark/                # >=2 solvers compared on the same axes / via metrics

Filename convention: every file inside one of these subfolders ends in
``_<folder-name>.<ext>`` (e.g. ``taylor_vs_analytic_benchmark.png``,
``exactbvp_reference.png``) so the artifact's role is unambiguous in
isolation. The lone exceptions are ``training/`` artifacts produced by
the trainer (``best_policy_<run-id>.pth``, ``history_<run-id>.csv``,
``loss_curve_<run-id>.png``) where the run-id timestamp already carries
the meaningful information.
"""

from __future__ import annotations

import os

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULTS_ROOT = os.path.join(_PKG_ROOT, "results")


def results_dir(*parts: str) -> str:
    """Return ``<package>/results/<parts...>``, creating it on demand.

    Parameters
    ----------
    *parts : str
        Path components below ``results/``. Typical usage::

            results_dir("LiquidationPortfolioOC", "training")
            results_dir("LiquidationPortfolioOC", "training", "training-plots")
            results_dir("LiquidationPortfolioOC", "benchmark")

    Returns
    -------
    str
        Absolute path to the directory; always exists when this returns.
    """
    p = os.path.join(_RESULTS_ROOT, *parts)
    os.makedirs(p, exist_ok=True)
    return p


def results_root() -> str:
    """Return the absolute path to ``<package>/results/`` itself."""
    os.makedirs(_RESULTS_ROOT, exist_ok=True)
    return _RESULTS_ROOT
