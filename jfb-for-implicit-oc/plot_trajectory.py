import torch
import matplotlib.pyplot as plt


def plot(traj_path, alphaG=1.0e3, sigma=0.02, title=None):
    data = torch.load(traj_path, map_location="cpu")

    t = data["t"].numpy()
    q = data["q"].numpy()
    S = data["S"].numpy()
    X = data["X"].numpy()
    u = data["u"].numpy()

    dt = float(t[1] - t[0])

    running_cost = 0.5 * (sigma ** 2) * ((q[:-1] ** 2).sum()) * dt
    terminal_penalty = alphaG * (q[-1] ** 2)
    cash_term = -X[-1]
    total_terminal = cash_term + terminal_penalty
    total_objective = running_cost + total_terminal

    avg_u = u.mean()
    max_u = u.max()
    q_initial = q[0]
    q_final = q[-1]
    liquidation_ratio = 1.0 - q_final / q_initial if q_initial != 0 else float("nan")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(t, q, color="tab:blue")
    axes[0, 0].set_title("Inventory q(t)")
    axes[0, 0].set_xlabel("t")
    axes[0, 0].set_ylabel("q")
    axes[0, 0].grid(True)

    axes[0, 1].plot(t[:-1], u, color="tab:orange")
    axes[0, 1].set_title("Trading rate u(t)")
    axes[0, 1].set_xlabel("t")
    axes[0, 1].set_ylabel("u")
    axes[0, 1].grid(True)

    axes[1, 0].plot(t, S, color="tab:green")
    axes[1, 0].set_title("Price S(t)")
    axes[1, 0].set_xlabel("t")
    axes[1, 0].set_ylabel("S")
    axes[1, 0].grid(True)
    axes[1, 0].ticklabel_format(style="plain", axis="y", useOffset=False)

    axes[1, 1].plot(t, X, color="tab:red")
    axes[1, 1].set_title("Cash X(t)")
    axes[1, 1].set_xlabel("t")
    axes[1, 1].set_ylabel("X")
    axes[1, 1].grid(True)

    metrics_text = (
        f"q(0) = {q_initial:.4f}\n"
        f"q(T) = {q_final:.4f}\n"
        f"Liquidated = {100*liquidation_ratio:.2f}%\n"
        f"X(T) = {X[-1]:.4f}\n"
        f"avg u = {avg_u:.4f}\n"
        f"max u = {max_u:.4f}\n"
        f"Running cost = {running_cost:.6f}\n"
        f"Terminal penalty = {terminal_penalty:.6f}\n"
        f"-X(T) = {cash_term:.6f}\n"
        f"Objective = {total_objective:.6f}"
    )

    fig.text(
        0.77, 0.20, metrics_text,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
    )

    if title is not None:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    plt.show()



if __name__ == "__main__":
    plot(r"C:\Users\fanch\Documents\AA_ETHZ\Semestre_2\ML_FIN\code\jfb-for-implicit-oc\results_LiquidationPortfolioOC\trajectory.pth",
         alphaG=1.0e3,
    sigma=0.02,
    title="Liquidation Policy"
    )