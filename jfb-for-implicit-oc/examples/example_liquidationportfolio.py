import torch, numpy as np, sys, time
import matplotlib as plt 
from LiquidationPortfolio import LiquidationPortfolioOC
from ImplicitNets            import Phi
from ImplicitNets            import ImplicitNetOC
from OptimalControlTrainer   import OptimalControlTrainer



def run_liquidation_jfb(config_oc: dict,
                        config_train: dict,
                        full_AD: bool = False,
                        device: str = "cpu",
                        plot_frequency=None):
    """
    Solves liquidation portfolio optimal problem with INN + JFB
    """

    print()
    print("####################################################################")
    print("##############                                        ##############")
    print("##############        Liquidation Portfolio with INN        ##############")
    print("##############                                        ##############")
    print("####################################################################")
    print()


    lp = LiquidationPortfolioOC(
        batch_size=64,
        t_initial=0.0,
        t_final=10.0,
        nt=100,
        sigma=0.02,
        kappa=1.0e-3,
        eta=0.1,
        gamma=2,
        epsilon=1.0e-3,
        alpha=20,
        q0_min=10.0,
        q0_max=15.0,
        S0=5.0,
        X0=0.0,
        device=device,
    )

    lp.track_all_fp_iters = full_AD     
    phi = Phi(3, 50, lp.state_dim, dev=device)
    inn = ImplicitNetOC(lp.state_dim, lp.control_dim,
                        alpha=1e-3, max_iters=200, tol=1e-4,
                        p_net=phi, oc_problem=lp,u_min=0,
                        u_max=10,use_control_limits=False,            
                        dev=device).to(device)

    opt       = torch.optim.Adam(inn.parameters(), lr=config_train["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt, mode="min", factor=0.5, patience=10)

    trainer = OptimalControlTrainer(inn, lp, opt, scheduler=scheduler,
                                    device=device)
    trainer.set_mode("standard")           # JFB = standard

    tag = "FullAD" if full_AD else "JFB"
    save_name = (f"best_policy_{tag}"
                 f"_Batch{config_oc['batch_size']}_"
                 f"{time.ctime().replace(' ','_').replace(':','_')}")

    z0 = lp.sample_initial_condition()
    trainer.train(z0,
                  num_epochs=config_train["epochs"],
                  plot_frequency=plot_frequency,
                  save_model_name=save_name)
    
    ###enregistrer les trajectoires pour plot
    with torch.no_grad():
        z0_plot = lp.sample_initial_condition()
        traj = lp.generate_trajectory(inn, z0_plot, lp.nt, return_full_trajectory=True)

    q = traj[0, 0, :].cpu()
    S = traj[0, 1, :].cpu()
    X = traj[0, 2, :].cpu()

    dt = (lp.t_final - lp.t_initial) / lp.nt
    t_grid = torch.linspace(lp.t_initial, lp.t_final, lp.nt + 1)

    u_vals = []
    with torch.no_grad():
        for i in range(lp.nt):
            z_i = traj[:, :, i]
            t_i = lp.t_initial + i * dt
            u_i = inn(z_i, t_i)
            u_vals.append(u_i[0, 0].item())

    u_vals = torch.tensor(u_vals)

    torch.save({
        "t": t_grid,
        "q": q,
        "S": S,
        "X": X,
        "u": u_vals,
        "z0": z0_plot.cpu(),
    }, f"results_LiquidationPortfolioOC/trajectory.pth")

    print(f"Trajectory saved to results_LiquidationPortfolioOC/trajectory_{save_name}.pth")

    




def main():
    seed = 420
    torch.manual_seed(seed);  np.random.seed(seed)

    config_oc = dict(batch_size=1,
                     nt=2,
                     t_final=2.0)        
    config_train = dict(lr=1e-3, epochs=20)

    device   = "cuda" if torch.cuda.is_available() else "cpu"
    n_trials   = 1                    
    # JFB
    for n in np.arange(n_trials):
        run_liquidation_jfb(config_oc, config_train,
                            full_AD=False, device=device)

if __name__ == "__main__":
    main()
