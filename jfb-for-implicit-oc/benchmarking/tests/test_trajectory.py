"""Tests for :mod:`benchmarking.trajectory`."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarking.trajectory import Trajectory


def _det_traj(N=11, state_dim=3, control_dim=1) -> Trajectory:
    t = np.linspace(0.0, 1.0, N)
    z = np.random.RandomState(0).randn(N, state_dim)
    u = np.random.RandomState(1).randn(N - 1, control_dim)
    return Trajectory(t=t, z=z, u=u, label="det", cost=1.0)


def _stoch_traj(n_paths=4, N=11, state_dim=3, control_dim=1) -> Trajectory:
    t = np.linspace(0.0, 1.0, N)
    z = np.random.RandomState(0).randn(n_paths, N, state_dim)
    u = np.random.RandomState(1).randn(n_paths, N - 1, control_dim)
    return Trajectory(t=t, z=z, u=u, label="stoch")


def test_deterministic_shape_properties():
    traj = _det_traj()
    assert not traj.is_stochastic
    assert traj.n_paths == 1
    assert traj.n_steps == 11
    assert traj.state_dim == 3
    assert traj.control_dim == 1
    assert traj.terminal_state().shape == (3,)
    assert traj.mean_path() is traj  # no-op for deterministic


def test_stochastic_shape_properties():
    traj = _stoch_traj(n_paths=5, N=11)
    assert traj.is_stochastic
    assert traj.n_paths == 5
    assert traj.n_steps == 11
    assert traj.terminal_state().shape == (5, 3)

    reduced = traj.mean_path()
    assert not reduced.is_stochastic
    assert reduced.z.shape == (11, 3)
    assert reduced.u.shape == (10, 1)
    assert reduced.meta["reduced_from_n_paths"] == 5


def test_shape_validation_rejects_mismatched_t_z():
    t = np.linspace(0, 1, 10)
    z = np.zeros((11, 3))
    with pytest.raises(ValueError):
        Trajectory(t=t, z=z)


def test_shape_validation_rejects_wrong_u_length():
    t = np.linspace(0, 1, 10)
    z = np.zeros((10, 3))
    u = np.zeros((7, 1))  # should be 9
    with pytest.raises(ValueError):
        Trajectory(t=t, z=z, u=u)


def test_shape_validation_rejects_mismatched_paths():
    t = np.linspace(0, 1, 10)
    z = np.zeros((3, 10, 2))
    u = np.zeros((4, 9, 1))  # wrong path count
    with pytest.raises(ValueError):
        Trajectory(t=t, z=z, u=u)


def test_none_u_allowed():
    t = np.linspace(0, 1, 5)
    z = np.zeros((5, 2))
    traj = Trajectory(t=t, z=z, u=None, label="noctrl")
    assert traj.u is None
    assert traj.control_dim == 0
