"""Tests for :class:`benchmarking.plotter.BenchmarkPlotter`."""

from __future__ import annotations

import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

from benchmarking.trajectory import Trajectory
from benchmarking.plotter import (
    BenchmarkPlotter,
    Panel,
    almgren_chriss_panels,
)


def _mk_traj(label, color, N=21) -> Trajectory:
    t = np.linspace(0.0, 1.0, N)
    z = np.stack(
        [
            np.linspace(1.0, 0.0, N),
            1.0 + 0.01 * t,
            np.linspace(0.0, 1.0, N),
        ],
        axis=1,
    )
    u = np.full((N - 1, 1), 0.5)
    return Trajectory(
        t=t, z=z, u=u, label=label, cost=float(-z[-1, 2]),
        style={"color": color, "lw": 2.0},
    )


def test_plotter_axis_count_matches_panels():
    trajs = [_mk_traj("A", "#1f77b4"), _mk_traj("B", "#d6604d")]
    plotter = BenchmarkPlotter(almgren_chriss_panels(), ncols=2)
    with tempfile.TemporaryDirectory() as tmp:
        fig = plotter.plot(trajs, save_path=os.path.join(tmp, "out.png"))
    assert len(fig.axes) == 4


def test_plotter_legend_has_one_entry_per_label():
    trajs = [_mk_traj("A", "#1f77b4"), _mk_traj("B", "#d6604d")]
    plotter = BenchmarkPlotter(almgren_chriss_panels(), ncols=2)
    with tempfile.TemporaryDirectory() as tmp:
        save = os.path.join(tmp, "legend.png")
        fig = plotter.plot(trajs, save_path=save)
    ax = fig.axes[0]
    _, labels = ax.get_legend_handles_labels()
    assert labels == ["A", "B"]


def test_plotter_rejects_empty_panel_list():
    with pytest.raises(ValueError):
        BenchmarkPlotter(panels=[], ncols=1)


def test_plotter_handles_single_trajectory():
    trajs = [_mk_traj("only", "#2ca02c")]
    plotter = BenchmarkPlotter(almgren_chriss_panels(), ncols=2)
    with tempfile.TemporaryDirectory() as tmp:
        plotter.plot(trajs, save_path=os.path.join(tmp, "single.png"))


def test_band_plot_falls_back_to_line_for_single_path():
    N = 10
    t = np.linspace(0, 1, N)
    z = np.random.RandomState(0).randn(N, 2)

    def extract(traj):
        return traj.t, traj.z[..., 0]

    traj = Trajectory(t=t, z=z, label="det", style={"color": "#444"})
    panels = [Panel("test", extract, "y", plot_type="band")]
    plotter = BenchmarkPlotter(panels, ncols=1)
    with tempfile.TemporaryDirectory() as tmp:
        plotter.plot([traj], save_path=os.path.join(tmp, "band_det.png"))


def test_band_plot_with_stochastic_trajectory():
    N, P = 15, 6
    t = np.linspace(0, 1, N)
    z = np.random.RandomState(0).randn(P, N, 2)

    def extract(traj):
        return traj.t, traj.z[..., 0]

    traj = Trajectory(t=t, z=z, label="stoch", style={"color": "#2166ac"})
    panels = [Panel("test", extract, "y", plot_type="band")]
    plotter = BenchmarkPlotter(panels, ncols=1)
    with tempfile.TemporaryDirectory() as tmp:
        fig = plotter.plot([traj], save_path=os.path.join(tmp, "band_stoch.png"))
    assert len(fig.axes) == 1


# ------------------------------------------------------------------
# Regression test for AlmgrenChrissBVPSolver
# ------------------------------------------------------------------

def test_almgren_bvp_matches_known_values():
    """Regression test against hardcoded numerical values.

    Protects the numerical output of :class:`AlmgrenChrissBVPSolver` from
    silent drift.  The hardcoded values were captured from the legacy
    implementation with the same problem parameters.
    """
    from LiquidationPortfolio import LiquidationPortfolioOC
    from benchmarking.solvers import AlmgrenChrissBVPSolver

    prob = LiquidationPortfolioOC(
        batch_size=1, t_initial=0.0, t_final=2.0, nt=100,
        sigma=0.02, kappa=1e-4, eta=0.1, gamma=2.0,
        epsilon=1e-2, alpha=30, q0_min=0.5, q0_max=1.5,
    )
    solver = AlmgrenChrissBVPSolver(prob)
    traj = solver.solve(np.array([1.0, 1.0, 0.0]))

    # Boundary conditions.
    np.testing.assert_allclose(traj.z[0, 0], 1.0, atol=1e-10)  # q(0) = q0
    np.testing.assert_allclose(traj.z[0, 1], 1.0, atol=1e-10)  # S(0) = S0
    np.testing.assert_allclose(traj.z[0, 2], 0.0, atol=1e-12)  # X(0) = 0

    # Hardcoded terminal values -- protect against silent drift.
    np.testing.assert_allclose(traj.z[-1, 0], -0.018364386004354263, atol=1e-8)
    np.testing.assert_allclose(traj.z[-1, 1],  0.9998981635613996,  atol=1e-8)
    np.testing.assert_allclose(traj.z[-1, 2],  0.9644591601663219,  atol=1e-8)
    np.testing.assert_allclose(traj.u[0, 0],   0.5078604023758793,  atol=1e-8)
    np.testing.assert_allclose(traj.cost,     -0.9543416399668142,  atol=1e-8)

    assert traj.label == "Exact BVP"
    assert traj.style.get("ls") == "--"
    assert traj.style.get("color") == "#2166ac"
