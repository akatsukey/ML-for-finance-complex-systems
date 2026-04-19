from ImplicitOC import ImplicitOC
import torch
from utils import GradientTester

class LiquidationPortfolioOC(ImplicitOC):
    """
    Optimal consumption-savings with habit formation.

    State z = [x, h_1, ..., h_m], control u = [u_1, ..., u_m].
    Dynamics:
        dx/dt = r x - sum_i u_i
        dh/dt = A * u^{\circ eta} - B * h^{\circ theta}
    Running cost: e^{-delta t} * sum_i (u_i - h_i)^{1-gamma} / (1-gamma)
    Terminal cost: epsilon * x(T)^{1-gamma} / (1-gamma)
    """

    def __init__(
        self,
        batch_size=64,
        t_initial=0.0,
        t_final=2.0,
        nt=100,
        sigma=0.02,
        kappa=1.0e-4,
        eta=0.1,
        gamma=2.0,
        epsilon=1.0e-2,
        alpha=30,
        q0_min=0.5,
        q0_max=1.5,
        S0=1.0,
        X0=0.0,
        device='cpu',
    ):
        state_dim = 3 #(q,S,X)
        control_dim = 1 #u
        super().__init__(state_dim, control_dim, batch_size,
                         t_initial, t_final, nt,alphaL=1.0, alphaG=1.0, device=device)
        self.oc_problem_name = "Liquidation Portfolio"

        self.epsilon=epsilon

        # finance params 
        self.sigma = sigma
        self.kappa = kappa
        self.eta = eta
        self.gamma = gamma
        self.alpha = alpha

        #initialisation values
        self.q0_min = q0_min
        self.q0_max = q0_max
        self.S0 = S0
        self.X0 = X0

    def compute_lagrangian(self, t, z, u):
        """
        Compute the running cost (Lagrangian).
        L(t, z, u) = 0.5 * sigma^2 * q^2

        Args:
            t (torch.Tensor or float): Current time
            z (torch.Tensor): State vector of shape (batch_size, state_dim) [q, S, X]
            u (torch.Tensor): Control input of shape (batch_size, control_dim) [trading rate u]

        Returns:
            torch.Tensor: Lagrangian values of shape (batch_size,)
        """

        if z.dim() == 1:
            z = z.unsqueeze(0)
            u = u.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        q = z[:,0]
        lag = 0.5*(self.sigma**2)*(q**2)
        return lag[0] if squeeze else lag

    def compute_grad_lagrangian(self, t, z, u):
        """
        Compute the gradient of the Lagrangian with respect to control.
        dL/du = 0

        Args:
            t (torch.Tensor or float): Current time
            z (torch.Tensor): State vector of shape (batch_size, state_dim) [q, S, X]
            u (torch.Tensor): Control input of shape (batch_size, control_dim) [trading rate u]            
        Returns:
            torch.Tensor: Gradient of Lagrangian of shape (batch_size, control_dim)
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
            u = u.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        grad=torch.zeros_like(u)

        return grad[0] if squeeze else grad

    def compute_f(self, t, z, u):
        """
        Compute the state dynamics.

        dq = -u
        dS = -kappa * u
        dX = S * u - eta * (u^2 + epsilon)^(gamma/2)

        Args:
            t (torch.Tensor or float): Current time
            z (torch.Tensor): State vector of shape (batch_size, state_dim) [q, S, X]
            u (torch.Tensor): Control input of shape (batch_size, control_dim) [trading rate u]            
        Returns:
            torch.Tensor: Derivative of z, shape (batch_size, state_dim)        
        """

        if z.dim() == 1:
            z = z.unsqueeze(0)
            u = u.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        q=z[:,0:1]
        S=z[:, 1:2]
        X=z[:, 2:3]

        dq=-u
        dS=-self.kappa*u
        #dX = S * u - self.eta * torch.abs(u).pow(self.gamma)
        dX = S * u - self.eta * (u.pow(2) + self.epsilon).pow(self.gamma / 2.0)


        result = torch.cat((dq, dS, dX), dim=1)
        return result[0] if squeeze else result

    def compute_grad_f_u(self, t, z, u):
        """
        Computes the gradient of the dynamics f with respect to the control u.

        Args:
            t (torch.Tensor or float): Current time 
            z (torch.Tensor): State vector of shape (batch_size, state_dim)
            u (torch.Tensor): Control vector of shape (batch_size, control_dim) 

        Return:
            torch.Tensor: Gradient of f with respect to u of shape (batch_size, control_dim, state_dim)
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
            u = u.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        batch = z.shape[0]
        grad = torch.zeros(batch, 1, 3, device=z.device)
        
        S=z[:,1:2]
        
        grad[:, 0, 0] = -1.0
        grad[:, 0, 1] = -self.kappa
        #grad[:, 0, 2] = (S-self.eta*self.gamma*torch.sign(u)*torch.abs(u).pow(self.gamma - 1)).squeeze(1)
        impact_grad = self.eta * self.gamma * u * (u.pow(2) + self.epsilon).pow(self.gamma / 2.0 - 1.0)
        grad[:, 0, 2] = (S - impact_grad).squeeze(1)

        return grad[0] if squeeze else grad

    def compute_grad_f_z(self, t, z, u):
        """
        Computes the gradient of the dynamics f with respect to the state vector z

        Args:
            t (torch.Tensor or float): Current time 
            z (torch.Tensor): State vector of shape (batch_size, state_dim)
            u (torch.Tensor): Control vector of shape (batch_size, control_dim) 

        Return:
            torch.Tensor: Gradient of f with respect to z of shape (batch_size, state_dim, state_dim)
        """
        
        if z.dim() == 1:
            z = z.unsqueeze(0)
            u = u.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        batch = z.shape[0]
        grad = torch.zeros(batch, 3, 3, device=z.device)

        grad[:,2,1]=u.squeeze(1)

        return grad[0] if squeeze else grad

    def compute_G(self, z):
        """
        G = -X(T) + alpha*(q(T)**2)
        Compute the terminal cost (without discounting).

        Args:
            z (torch.Tensor): State vector of shape (batch_size, state_dim)
            
        Returns:
            torch.Tensor: Terminal cost values of shape (batch_size,)
        """

        if z.dim() == 1:
            z = z.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        q = z[:, 0]
        X = z[:, 2]
        G = -X+self.alpha*(q**2)
        return G[0] if squeeze else G


    def compute_grad_G_z(self, z):
        """
        Computes the gradient of the terminal cost G with respect to the state vector z

        Args:
            t (torch.Tensor or float): Current time 
            z (torch.Tensor): State vector of shape (batch_size, state_dim)
            u (torch.Tensor): Control vector of shape (batch_size, control_dim) 

        Return:
            torch.Tensor: Gradient of G with respect to z of shape (batch_size, state_dim)
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        batch = z.shape[0]
        grad = torch.zeros(batch, 3, device=z.device)

        q=z[:,0]
        grad[:, 0] = 2.0 * self.alpha * q
        grad[:, 2] = -1.0
        
        return grad[0] if squeeze else grad

    def sample_initial_condition(self):
        q0 = self.q0_min + (self.q0_max - self.q0_min) * torch.rand(
        self.batch_size, 1, device=self.device)

        S0= torch.full((self.batch_size, 1), self.S0, device=self.device)
        X0 = torch.full((self.batch_size, 1), self.X0, device=self.device)
        return torch.cat((q0,S0,X0), dim=1).to(self.device)

    def generate_trajectory(self, u, z0, nt, return_full_trajectory=False):
        batch = z0.shape[0]
        D = self.state_dim

        traj = torch.zeros(batch, D, nt+1, device=z0.device)
        traj[:, :, 0] = z0
        dt = (self.t_final - self.t_initial) / nt
        t = self.t_initial

        for i in range(nt):
            if torch.is_tensor(u):
                curr = u[:, :, i]
            else:
                curr = u(traj[:, :, i], t)
            traj[:, :, i+1] = traj[:, :, i] + dt * self.compute_f(t, traj[:, :, i], curr)
            t += dt
        return traj if return_full_trajectory else traj[:, :, -1]

# Example usage
if __name__ == "__main__":

    device = 'cpu'
    batch_size = 10
    nt = 100

    prob = LiquidationPortfolioOC(batch_size=batch_size,
                                    t_initial=0.0,
                                    t_final=1.0,
                                    nt=nt,
                                    sigma=0.02,
                                    kappa=1.0e-4,
                                    eta=0.1,
                                    gamma=2.0,
                                    q0_min=0.5,
                                    q0_max=1.5,
                                    S0=1.0,
                                    X0=0.0,
                                    device=device)
                                    
    u_rand = torch.randn(batch_size, 1, nt, device=device)


    # Gradient tests
    test_z = torch.tensor([[1.0, 1.0, 0.0], [0.8, 1.1, 0.1]], dtype=torch.float32)
    test_u = torch.tensor([[0.1], [0.2]], dtype=torch.float32)

    test_z = test_z.repeat(batch_size // 2, 1).to(device)
    test_u = test_u.repeat(batch_size // 2, 1).to(device)

    print("Running gradient tests...")
    GradientTester.run_all_tests(prob, test_z, test_u)
