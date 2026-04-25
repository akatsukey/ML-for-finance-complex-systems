#!/usr/bin/env python3
"""
Plot JFB (implicit-net) rollouts for :class:`LiquidationPortfolioOC`.

Run **from the repository** (either cwd = ``jfb-for-implicit-oc`` or project root):

    cd jfb-for-implicit-oc
    python plot_liquidation_jfb.py

    # or
    python jfb-for-implicit-oc/plot_liquidation_jfb.py

**Default behaviour:** trains a fresh policy for ``--train-epochs`` (small run),
then saves a six-panel figure (JFB vs exact BVP when ``γ=2``) via
:class:`liquidation_benchmark.LiquidationBenchmark`.

**Use a checkpoint** saved by :class:`OptimalControlTrainer` (``state_dict`` only):

    python plot_liquidation_jfb.py --checkpoint results/LiquidationPortfolioOC/training/best_policy_JFB_<run_id>.pth

You must use the **same** ``LiquidationPortfolioOC`` hyperparameters as training
when loading a checkpoint (defaults below match ``liquidation_benchmark`` smoke
settings so the exact reference lines up).
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

# Local imports: script may be run from project root or this directory.
# core/ and models/ still use flat imports, so they need to be on sys.path
# in addition to the package root (which is needed for `core.paths`, etc.).
_ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in (_ROOT, os.path.join(_ROOT, "core"), os.path.join(_ROOT, "models")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ImplicitNets import ImplicitNetOC, Phi
from LiquidationPortfolio import LiquidationPortfolioOC
from OptimalControlTrainer import OptimalControlTrainer
from liquidation_benchmark import LiquidationBenchmark, benchmark_png_path
from core.paths import results_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to policy .pth (torch.save(state_dict)). If omitted, train with --train-epochs.",
    )
    p.add_argument("--train-epochs", type=int, default=40, help="Adam epochs when no checkpoint (default: 40).")
    p.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate when training.")
    p.add_argument("--batch-size", type=int, default=64, help="Training batch size / sample_initial_condition size.")
    p.add_argument("--n-show", type=int, default=5, help="Max trajectories overlaid in the figure.")
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PNG path. Default: results/<ProblemClassName>/benchmark/jfb_vs_exactbvp_benchmark.png",
    )
    p.add_argument("--device", type=str, default=None, help="cpu or cuda (default: auto).")
    p.add_argument(
        "--tag",
        type=str,
        default="JFB",
        help="Run tag passed to OptimalControlTrainer (becomes part of every artifact filename).",
    )
    # Problem parameters (must match training when using --checkpoint)
    p.add_argument("--t-final", type=float, default=2.0)
    p.add_argument("--nt", type=int, default=100)
    p.add_argument("--sigma", type=float, default=0.02)
    p.add_argument("--kappa", type=float, default=1e-4)
    p.add_argument("--eta", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument("--epsilon", type=float, default=1e-2)
    p.add_argument("--alpha", type=float, default=30.0)
    p.add_argument("--q0-min", type=float, default=0.5)
    p.add_argument("--q0-max", type=float, default=1.5)
    p.add_argument("--S0", type=float, default=1.0)
    p.add_argument("--X0", type=float, default=0.0)
    return p.parse_args()


def build_problem(args: argparse.Namespace, device: str) -> LiquidationPortfolioOC:
    return LiquidationPortfolioOC(
        batch_size=args.batch_size,
        t_initial=0.0,
        t_final=args.t_final,
        nt=args.nt,
        sigma=args.sigma,
        kappa=args.kappa,
        eta=args.eta,
        gamma=args.gamma,
        epsilon=args.epsilon,
        alpha=args.alpha,
        q0_min=args.q0_min,
        q0_max=args.q0_max,
        S0=args.S0,
        X0=args.X0,
        device=device,
    )


def build_policy(prob: LiquidationPortfolioOC, device: str) -> ImplicitNetOC:
    phi = Phi(3, 50, prob.state_dim, dev=device)
    return ImplicitNetOC(
        prob.state_dim,
        prob.control_dim,
        alpha=1e-3,
        max_iters=200,
        tol=1e-4,
        p_net=phi,
        oc_problem=prob,
        u_min=0.0,
        u_max=10.0,
        use_control_limits=True,
        dev=device,
    ).to(device)


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    prob = build_problem(args, device)
    inn = build_policy(prob, device)

    if args.checkpoint:
        ckpt = os.path.abspath(args.checkpoint)
        if not os.path.isfile(ckpt):
            raise SystemExit(f"Checkpoint not found: {ckpt}")
        state = torch.load(ckpt, map_location=device)
        inn.load_state_dict(state, strict=True)
        print(f"Loaded policy weights from: {ckpt}")
    else:
        if args.train_epochs <= 0:
            raise SystemExit("Provide --checkpoint or set --train-epochs > 0.")
        opt = torch.optim.Adam(inn.parameters(), lr=args.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=8
        )
        trainer = OptimalControlTrainer(
            inn, prob, opt, scheduler=sched, device=device, tag=args.tag,
        )
        trainer.set_mode("standard")
        z0_train = prob.sample_initial_condition()
        trainer.train(
            z0_train,
            num_epochs=args.train_epochs,
            verbose=True,
            plot_frequency=10,
        )
        # The trainer already reloaded the best checkpoint into `inn` during
        # finalize(). Keep the path around in case downstream tooling wants it.
        best_path = trainer.run_io.policy_path()
        if os.path.isfile(best_path):
            print(f"Best policy stored at: {os.path.abspath(best_path)}")

    inn.eval()
    bench = LiquidationBenchmark(prob)
    z0_plot = prob.sample_initial_condition()

    out = args.output or os.path.join(
        results_dir(type(prob).__name__, "benchmark"),
        "jfb_vs_exactbvp_benchmark.png",
    )
    bench.plot_comparison(inn, z0_plot, save_path=out, n_show=args.n_show)
    print(f"Figure written to: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
