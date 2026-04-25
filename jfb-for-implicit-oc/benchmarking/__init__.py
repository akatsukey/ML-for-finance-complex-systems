"""
benchmarking
------------
Reusable benchmarking harness for :class:`ImplicitOC` problems.

The package decouples *reference solvers* (which produce a
:class:`Trajectory`) from *plotting* (:class:`BenchmarkPlotter`) and
*metrics* (:mod:`benchmarking.metrics`), so that new problem classes can
plug in by writing only a new solver and, optionally, a new panel-set
factory.  See :mod:`benchmarking.solvers` for a stubbed multi-asset
example.
"""

from __future__ import annotations

from .paths import benchmark_png_dir, benchmark_png_path
from .trajectory import Trajectory
from .solvers import (
    ReferenceSolver,
    AlmgrenChrissBVPSolver,
    JFBPolicyRollout,
)
from .plotter import (
    Panel,
    BenchmarkPlotter,
    almgren_chriss_panels,
)
from . import metrics
from . import gradient_checks

__all__ = [
    "Trajectory",
    "ReferenceSolver",
    "AlmgrenChrissBVPSolver",
    "JFBPolicyRollout",
    "BenchmarkPlotter",
    "Panel",
    "almgren_chriss_panels",
    "benchmark_png_dir",
    "benchmark_png_path",
    "metrics",
    "gradient_checks",
]
