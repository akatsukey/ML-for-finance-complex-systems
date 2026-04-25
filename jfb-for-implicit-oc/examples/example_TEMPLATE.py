"""
Template runner for a new optimal-control problem.

Copy-paste workflow
~~~~~~~~~~~~~~~~~~~

1. Copy this file to ``examples/example_<myproblem>.py``.
2. Resolve **TODO[1]**: import your concrete subclass of :class:`ImplicitOC`
   from ``models/``.
3. Resolve **TODO[2]**: fill in the constructor call with your concrete
   hyperparameters (dynamics, costs, IC distribution). Make sure
   ``state_dim`` / ``control_dim`` (set inside the class's ``__init__``)
   match what your dynamics expect.
4. (Optional) tweak ``Phi`` width, ``ImplicitNetOC`` control limits,
   optimizer, LR scheduler, ``num_epochs``, ``plot_frequency`` if the
   defaults below are not appropriate.
5. Run it::

       cd jfb-for-implicit-oc
       python examples/example_<myproblem>.py

   The trainer writes the canonical six-artifact bundle under
   ``results/<MyProblemOC>/`` automatically — there is nothing else for
   this file to do.

Strict invariants this file MUST preserve
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(See the "Smell test" section in the project README.)

* No ``os.path.join`` calls.
* No ``save_path=`` literals.
* No ``save_name`` arguments.
* No ``matplotlib`` code.

If you find yourself adding any of the above, the right fix is to extend
:class:`core.run_io.RunIO`, not to bypass it from the runner.
"""

import os
import sys

import numpy as np
import torch

# --- sys.path bootstrap (lets the script run regardless of cwd) -----------
# core/ and models/ still use flat imports (e.g. `from ImplicitOC import ...`),
# so they need to be on sys.path themselves; the project root is needed for
# `core.paths`, `core.run_io`, etc.
_HERE = os.path.dirname(os.path.abspath(__file__))           # .../jfb-for-implicit-oc/examples
_ROOT = os.path.dirname(_HERE)                               # .../jfb-for-implicit-oc
for _p in (_ROOT, os.path.join(_ROOT, "core"), os.path.join(_ROOT, "models")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# TODO[1]: import your concrete problem class from models/.
# from MyProblem            import MyProblemOC                # models/MyProblem.py

from ImplicitNets          import Phi, ImplicitNetOC          # core/
from OptimalControlTrainer import OptimalControlTrainer       # core/


def run_template(
    *,
    full_AD: bool = False,
    epochs: int = 20,
    lr: float = 1e-3,
    plot_frequency: int = 5,
    device: str = "cpu",
) -> OptimalControlTrainer:
    """Train a JFB policy on the chosen problem and return the trainer.

    The trainer writes its full six-artifact bundle under
    ``results/<MyProblemOC>/`` — see the README for the canonical layout.
    """

    print()
    print("####################################################################")
    print("##############                                        ##############")
    print("##############     <MyProblemOC> with INN             ##############")
    print("##############                                        ##############")
    print("####################################################################")
    print()

    # ------------------------------------------------------------------ #
    # TODO[2]: instantiate your problem.                                 #
    # ------------------------------------------------------------------ #
    # prob = MyProblemOC(
    #     batch_size=64,
    #     t_initial=0.0,
    #     t_final=...,
    #     nt=...,
    #     # ... model hyperparameters (dynamics, costs, IC distribution)
    #     device=device,
    # )
    raise NotImplementedError(
        "example_TEMPLATE.py: fill in TODO[1] (import) and TODO[2] "
        "(constructor) above, then delete this raise."
    )
    # ------------------------------------------------------------------ #

    prob.track_all_fp_iters = full_AD

    # Network: defaults are sensible for low-dimensional problems
    # (state_dim, control_dim ≲ 10). Increase the hidden width on harder
    # problems; relax / tighten control limits to match your dynamics.
    phi = Phi(3, 50, prob.state_dim, dev=device)
    inn = ImplicitNetOC(
        prob.state_dim, prob.control_dim,
        alpha=1e-3, max_iters=200, tol=1e-4,
        p_net=phi, oc_problem=prob,
        u_min=0, u_max=10, use_control_limits=True,
        dev=device,
    ).to(device)

    opt = torch.optim.Adam(inn.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=10,
    )

    tag = "FullAD" if full_AD else "JFB"
    trainer = OptimalControlTrainer(
        inn, prob, opt, scheduler=scheduler, device=device, tag=tag,
    )
    trainer.set_mode("standard")

    z0 = prob.sample_initial_condition()
    trainer.train(z0, num_epochs=epochs, plot_frequency=plot_frequency)
    return trainer


def main():
    seed = 420
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_template(full_AD=False, epochs=20, lr=1e-3,
                 plot_frequency=5, device=device)


if __name__ == "__main__":
    main()
