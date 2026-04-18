"""
Working Jacobian-based trainer that extends your existing ImplicitNetOC.
This follows the exact pattern from your utils_JFB.py but adapted for optimal control.
"""

import torch
import torch.nn as nn
import time
import os
import pandas as pd
from matplotlib import pyplot as plt
from Quadcopter import QuadcopterOC
from CVXPolicy import CVXPolicy_MC, CVXPolicy_Quadcopter
from ImplicitNets import ImplicitNetOC
## for optimal consumption example, use --
# from ImplicitNets import ImplicitNetOC_pos as ImplicitNetOC
import psutil
import copy

class LRScheduler:
    """
    Custom LR scheduler class that is similar to PyTorch's
    ReduceLROnPlateau LR scheduler, but reduces the learning rate
    by some factor after a fixed number of consecutive epochs during 
    which there is no decrease in the loss function. ReduceLROnPlateau 
    reduces the learing rate only after a fixed number of epochs in which 
    the loss function is less than the best loss achieved up to that point
    """

    def __init__(self, optim, init_lr, min_lr, fact, pat):
        self.optimizer = optim
        self.initial_lr = init_lr
        self.min_lr = min_lr
        self.factor = fact
        self.patience = pat
        self.current_lr = init_lr
        self.num_no_decr = 0
        self.prev_loss = float('inf')
        self.current_epoch = 0

    def get_initial_lr(self):
        return self.initial_lr

    def get_current_lr(self):
        return self.current_lr

    def get_current_epoch(self):
        return self.current_epoch

    def get_prev_loss(self):
        return self.get_prev_loss

    def step(self, new_loss):
        self.current_epoch += 1

        if new_loss < self.prev_loss:
            self.num_no_decr = 0
        else:
            self.num_no_decr += 1
        self.prev_loss = new_loss

        if (self.num_no_decr > self.patience) and (self.current_epoch != 1):
            self.current_lr *= self.factor

            if self.current_lr < self.min_lr:
                self.current_lr = self.min_lr

            for g in self.optimizer.param_groups:
                g['lr'] = self.current_lr
                
            self.num_no_decr = 0


class OptimalControlTrainer:
    def __init__(self, policy_net, oc_problem, optimizer, scheduler=None, ver=False, device='cpu'):
        self.policy = policy_net
        self.oc_problem = oc_problem
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.mode = 'standard' # Default mode
        self.verify = ver # Whether or not to do additional computations to verify theoretical assumptions, default is False
        # Gradient clipping for direct control methods
        self.enable_grad_clip = False
        self.grad_clip_value = 1.0
        # === CHANGE: Restored full history tracking ===
        self.history = {k: [] for k in [
            'epoch', 'loss', 'running_cost', 'terminal_cost', 
            'cHJB', 'cHJBfin', 'cadj', 'cadjfin',
            'time_per_epoch', 'grad_norm', 'lr', 'max_fp_itrs',
            'max_fp_res_norm', 'memory_MB', 'max_memory_MB',
            'gpu_memory_MB', 'gpu_max_memory_MB', 'work_units',
            'max_grad_H', 'avg_grad_H', 'smallest_M_sdval', 
            'largest_M_sdval', 'smallest_lambda_min', 'largest_lambda_max',
            'max_grad_T_u', 'avg_grad_T_u', 'sd_grad_T_u', 'angle'
        ]}

    def set_mode(self, mode='standard'):
        if mode not in ['standard', 'cvx']:
            raise ValueError("Mode must be 'standard' or 'cvx'")
        self.mode = mode
        print(f"Trainer mode set to '{self.mode}'")

    def standard_step(self, z0):
        self.policy.train()
        max_fp_itrs = 0.0
        max_fp_res_norm = 0.0
        
        if self.mode == 'standard':
            convergence_stats = self.policy.get_convergence_stats()
            max_fp_itrs = convergence_stats['fp_depth']
            max_fp_res_norm = convergence_stats['max_res_norm']

        self.optimizer.zero_grad()
        # === CHANGE: Unpack all 7 values from compute_loss ===
        total_cost, run_cost, term_cost, cHJB, cHJBfin, cadj, cadjfin, max_grad_H, avg_grad_H = self.oc_problem.compute_loss(self.policy, z0)
        total_cost.backward()
        if self.enable_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_value)
        self.optimizer.step()
        # === CHANGE: Return all cost components ===
        return {
            'loss': total_cost.item(), 
            'running_cost': run_cost.item(), 
            'terminal_cost': term_cost.item(),
            'cHJB': cHJB.item(),
            'cHJBfin': cHJBfin.item(),
            'cadj': cadj.item(),
            'cadjfin': cadjfin.item(),
            'max_fp_itrs': max_fp_itrs,
            'max_fp_res_norm': max_fp_res_norm,
            'lr': self.scheduler.get_last_lr()[0],
            'max_grad_H': max_grad_H,
            'avg_grad_H': avg_grad_H
        }
    
    def standard_step_verify(self, z0):
        self.policy.train()
        max_fp_itrs = 0.0
        max_fp_res_norm = 0.0
        angle = 0.0
        
        if self.mode == 'standard':
            convergence_stats = self.policy.get_convergence_stats()
            max_fp_itrs = convergence_stats['fp_depth']
            max_fp_res_norm = convergence_stats['max_res_norm']

        self.optimizer.zero_grad()

        # Copy weights to compute gradient using full AD as well as JFB for angle computation
        #policy_AD = copy.deepcopy(self.policy) 
        # === CHANGE: Unpack all values from compute_loss ===
        total_cost, run_cost, term_cost, cHJB, cHJBfin, cadj, cadjfin, max_grad_H, avg_grad_H, smallest_M_sdval, largest_M_sdval, smallest_lambda_min, largest_lambda_max, max_grad_T_u, avg_grad_T_u, sd_grad_T_u = self.oc_problem.compute_loss_verify(self.policy, z0)
        #og_full_AD = self.oc_problem.track_all_fp_iters
        #self.oc_problem.track_all_fp_iters = True
        #total_cost_ad, run_cost_ad, term_cost_ad, cHJB_ad, cHJBfin_ad, cadj_ad, cadjfin_ad, max_grad_H_ad, avg_grad_H_ad = self.oc_problem.compute_loss(policy_AD, z0)
        #self.oc_problem.track_all_fp_iters = og_full_AD
        total_cost.backward()
        if self.enable_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_value)
        #total_cost_ad.backward()
        """
        nabla_theta_J_ = []
        for p_ad in policy_AD.parameters():
            if p_ad.grad is not None:
                nabla_theta_J_.append(p_ad.grad.view(-1))
        nabla_theta_J = torch.cat(nabla_theta_J_)
        d_JFB_ = []
        for p in self.policy.parameters():
            if p.grad is not None:
                d_JFB_.append(p.grad.view(-1))
        d_JFB = torch.cat(d_JFB_)
        angle = torch.acos(torch.dot(nabla_theta_J, d_JFB)/(torch.linalg.norm(nabla_theta_J, ord=2)*torch.linalg.norm(d_JFB, ord=2))).item()
        print(f"angle between true gradient and JFB approximation: {angle:.4e}")
        # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        """
        self.optimizer.step()
        # === CHANGE: Return all cost components ===
        return {
            'loss': total_cost.item(), 
            'running_cost': run_cost.item(), 
            'terminal_cost': term_cost.item(),
            'cHJB': cHJB.item(),
            'cHJBfin': cHJBfin.item(),
            'cadj': cadj.item(),
            'cadjfin': cadjfin.item(),
            'max_fp_itrs': max_fp_itrs,
            'max_fp_res_norm': max_fp_res_norm,
            'lr': self.scheduler.get_last_lr()[0],
            'max_grad_H': max_grad_H,
            'avg_grad_H': avg_grad_H,
            'smallest_M_sdval': smallest_M_sdval,
            'largest_M_sdval': largest_M_sdval,
            'smallest_lambda_min': smallest_lambda_min,
            'largest_lambda_max': largest_lambda_max,
            'max_grad_T_u': max_grad_T_u,
            'avg_grad_T_u': avg_grad_T_u,
            'sd_grad_T_u': sd_grad_T_u,
            'angle': angle
        }

    def cvx_step(self, z0):
        # The logic is identical for CVX mode, but we keep it separate for clarity
        return self.standard_step(z0)

    def train_epoch(self, z0):
        if self.mode == 'cvx':
            return self.cvx_step(z0)
        elif self.mode == 'standard' and self.verify: # verify assumptions
            return self.standard_step_verify(z0)
        else: # standard mode
            return self.standard_step(z0)
    
    def train(self, z0, num_epochs, verbose=True, plot_frequency=25, save_model_name='best_policy'):
        oc_name = type(self.oc_problem).__name__
        save_dir = f'results_{oc_name}/{self.mode}_mode'
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f'{save_model_name}.pth')
        history_path = os.path.join(save_dir, f'history_{save_model_name}.csv')
        print(f"Starting training in '{self.mode}' mode for {num_epochs} epochs. Results in '{save_dir}'")
        print("-" * 60)
        best_loss = float('inf')

        process = psutil.Process(os.getpid())
        for epoch in range(1, num_epochs + 1):
            # GPU Memory usage (in MB)
            gpu_memory_MB = 0.0
            gpu_max_memory_MB = 0.0
            max_memory_MB = 0.0
            epoch_start_time = time.time()
            step_info = self.train_epoch(z0)
            self.scheduler.step(step_info['loss'])
            
            memory_MB = process.memory_info().rss / 1024 / 1024
            if memory_MB > max_memory_MB:
                max_memory_MB = memory_MB
            if torch.cuda.is_available():
                gpu_memory_MB = torch.cuda.memory_allocated() / 1024 / 1024
                gpu_max_memory_MB = torch.cuda.max_memory_allocated() / 1024 / 1024
            time_per_epoch = time.time() - epoch_start_time

            grad_norm = sum(p.grad.norm().item()**2 for p in self.policy.parameters() if p.grad is not None)**0.5

            work_units = self.oc_problem.batch_size*self.policy.tracked_iters
            if (step_info['max_fp_itrs'] < self.policy.tracked_iters) or self.oc_problem.track_all_fp_iters:
                work_units = self.oc_problem.batch_size*step_info['max_fp_itrs']
            

            # This loop correctly populates the full history dict
            for key in self.history:
                if key == 'memory_MB':
                    self.history[key].append(memory_MB)
                elif key == 'max_memory_MB':
                    self.history[key].append(max_memory_MB)
                elif key == 'gpu_memory_MB':
                    self.history[key].append(gpu_memory_MB)
                elif key == 'gpu_max_memory_MB':
                    self.history[key].append(gpu_max_memory_MB)
                elif key == 'work_units':
                    self.history[key].append(work_units)
                else:
                    self.history[key].append(locals().get(key, step_info.get(key, 0)))

            # === CHANGE: Updated print statement to show HJB/ADJ costs and memory ===
            if verbose:
                print(f"Epoch {epoch:03d} | Loss: {step_info['loss']:.3e} | L: {step_info['running_cost']:.3e} | G: {step_info['terminal_cost']:.3e} | "
                        f"HJB: {step_info.get('cHJB', 0):.3e} | HJB fin: {step_info.get('cHJBfin',0):.3e} |Adj: {step_info.get('cadj', 0):.2e} | "
                      f"Grad: {grad_norm:.2e} | Time: {time_per_epoch:.2f}s | "
                      f"CPU Mem: {memory_MB:.1f}MB | Max CPU: {max_memory_MB:.1f}MB | "
                      f"GPU Mem: {gpu_memory_MB:.1f}MB | Max GPU: {gpu_max_memory_MB:.1f}MB | lr: {step_info['lr']:.3e} | "
                      f"max_fp_itrs: {step_info['max_fp_itrs']} | res_norm: {step_info['max_fp_res_norm']:.3e} | max_grad_H: {step_info['max_grad_H']:.3e} | avg_grad_H: {step_info['avg_grad_H']:.3e} ")

            prev_loss = step_info['loss']

            if step_info['loss'] < best_loss:
                best_loss = step_info['loss']
                torch.save(self.policy.state_dict(), save_path)
                if verbose: print(f"    -> New best model saved to {save_path} with loss {best_loss:.4e}")

            if plot_frequency and epoch % plot_frequency == 0:
                z_traj = self.oc_problem.generate_trajectory(self.policy, z0,
                                                             self.oc_problem.nt, return_full_trajectory=True)
                traj_plot_dir = os.path.join(save_dir, 'traj_plots')
                os.makedirs(traj_plot_dir, exist_ok=True)
                traj_plot_path = os.path.join(traj_plot_dir, f'traj_{save_model_name}_epoch_{epoch:04d}.png')
                self.oc_problem.plot_position_trajectories(z_traj.detach(), save_path=traj_plot_path)
                if verbose:
                    print(f"    -> Trajectory plot saved to {traj_plot_path}")

            # === Save CSV after each epoch ===
            pd.DataFrame(self.history).to_csv(history_path, index=False)
        #self.plot_loss_curve(save_dir, save_model_name)
        return self.history
    
    def plot_loss_curve(self, save_dir, save_model_name):
        plt.figure(figsize=(10, 6)); plt.yscale('log'); plt.grid(True)
        plt.plot(self.history['epoch'], self.history['loss'], label='Total Loss')
        plt.title(f'Training Loss (Mode: {self.mode})'); plt.legend()
        plt.savefig(os.path.join(save_dir, f'loss_curve_{save_model_name}.png')); plt.close()

if __name__ == '__main__':

    # --- Shared Configuration ---
    config_OC = {'batch_size': 10, 'nt': 2, 't_final': 10.0, }
    config_train = { 'lr': 1e-3, 'epochs': 2}
    device = 'cpu'

    # --- 1. Train with Standard Mode ---
    print("\n\n--- Testing Standard Mode ---")
    mc_problem_std = MountainCarOC(device=device, **config_OC)
    implicit_net = ImplicitNetOC(
        mc_problem_std.state_dim, mc_problem_std.control_dim, oc_problem=mc_problem_std, dev=device
    ).to(device)
    optimizer_std = torch.optim.Adam(implicit_net.parameters(), lr=config_train['lr'])

    trainer_std = OptimalControlTrainer(implicit_net, mc_problem_std, optimizer_std, device=device)
    trainer_std.set_mode('standard')
    z0_std = mc_problem_std.sample_initial_condition()
    trainer_std.train(z0_std, num_epochs=config_train['epochs'], save_model_name='best_standard_policy')
    
    # --- 2. Train with CVXPY Mode ---
    print("\n--- Testing CVXPY Mode ---")
    from ImplicitNets import Phi
    mc_problem = MountainCarOC(device=device, **config_OC)
    Phi = Phi(3, 10, mc_problem.state_dim)
    cvx_policy = CVXPolicy_MC(mc_problem.state_dim, mc_problem.control_dim,
                            mc_problem.power, p_net=Phi).to(device)
    optimizer = torch.optim.Adam(cvx_policy.parameters(), lr=config_train['lr'])
    
    trainer_cvx = OptimalControlTrainer(cvx_policy, mc_problem, optimizer, device=device)
    trainer_cvx.set_mode('cvx')
    z0 = mc_problem.sample_initial_condition()
    trainer_cvx.train(z0, num_epochs=config_train['epochs'], save_model_name='best_cvx_policy')


    # check grad_u H of the trained model
    print("\n--- Checking grad_u H of the trained model ---")
    
    
