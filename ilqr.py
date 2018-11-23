import sys
import autograd.numpy as np
from autograd import grad, jacobian

"""
Regularization for inverted eigenvalues taken from https://studywolf.wordpress.com/2016/02/03/the-iterative-linear-quadratic-regulator-method/

"""
class iLQR():
    def __init__(self, system, cost, max_epoch=100, convergence_threshold = 0.0001):
        self.system = system
        self.cost = cost

        self.lamb = 1.0           #  regularization term for inverted eigenvalues
        self.lamb_factor = 10     
        self.max_lamb = 10000000

        self.max_epoch = max_epoch
        self.convergence_threshold = convergence_threshold
    
    def set_max_epoch(self, max_epoch):
        self.max_epoch = max_epoch

    def gradient_calculations(self):
        F = self.system.dynamics
        Fx = jacobian(F, 0)
        Fu = jacobian(F, 1)
        Fxx = jacobian(Fx, 0)
        Fuu = jacobian(Fu, 1)
        Fux = jacobian(Fu, 0)

        L = self.cost.running_cost
        Lx = grad(L, 0)
        Lu = grad(L, 1)
        Lxx = jacobian(Lx, 0) # Lxx = hessian(L, 0) 
        Lux = jacobian(Lu, 0) 
        Luu = jacobian(Lu, 1)

        Lf = self.cost.terminal_cost
        Lfx = grad(Lf)
        Lfxx = jacobian(Lfx)

        return Fx, Fu, Fxx, Fuu, Fux, Lx, Lu, Lxx, Lux, Luu, Lf, Lfx, Lfxx
    
    def initial_rollout(self, x_init, U):
        """
        perfrom forward roll-out for random control sequence
        """   
        trajectory = [x_init]
        for t in range(len(U)-1):
            next_state = self.system.dynamics(trajectory[t], U[t])
            trajectory.append(next_state)

        return trajectory
    
    def get_trajectory_cost(self, trajectory, control_sequence):
        cost = 0
        for t in range(len(control_sequence)-1):
            cost += self.cost.running_cost(trajectory[t], control_sequence[t])
        cost += self.cost.terminal_cost(trajectory[t])

        return cost
    
    def forward_rollout(self, trajectory, control_sequence, K, k):
        new_trajectory = np.zeros_like(trajectory)
        new_control_sequence = np.zeros_like(control_sequence)
        new_trajectory[0] = trajectory[0]

        for t in range(len(control_sequence)-1):
 
            new_control_sequence[t] = control_sequence[t] + k[t] + K[t] @ (new_trajectory[t] - trajectory[t])
            new_trajectory[t+1] = self.system.dynamics(new_trajectory[t], new_control_sequence[t])

        return new_trajectory, new_control_sequence

    def backward_pass(self, trajectory, control_sequence):
        Fx, Fu, Fxx, Fuu, Fux, Lx, Lu, Lxx, Lux, Luu, Lf, Lfx, Lfxx = self.gradient_calculations()
        V = self.cost.terminal_cost(trajectory[-1])
        Vx = Lfx(trajectory[-1]) # Vx = Lx(trajectory[-1])
        Vxx = Lfxx(trajectory[-1]) # Vxx = Lxx(trajectory[-1])

        K = [None] * len(control_sequence) 
        k = [None] * len(control_sequence)

        for t in range(len(trajectory)-2, -1, -1):

            Fxt = Fx(trajectory[t], control_sequence[t])
            Fut = Fu(trajectory[t], control_sequence[t])
            Lxt = Lx(trajectory[t], control_sequence[t])
            Lxxt = Lxx(trajectory[t], control_sequence[t])
            Lut = Lu(trajectory[t], control_sequence[t])
            Luxt = Lux(trajectory[t], control_sequence[t])
            Luut = Luu(trajectory[t], control_sequence[t])
            
            #if dt exists 
            # Fxt *= self.system.dt
            # Fxt += np.eye(self.system.state_dim) 
            # Fut *= self.system.dt
            # Lxt *= self.system.dt
            # Lxxt *= self.system.dt
            # Lut *= self.system.dt
            # Luxt *= self.system.dt
            # Luut *= self.system.dt
            
            Qx = Lxt + Fxt.T @ Vx
            Qu = Lut + Fut.T @ Vx
            Qxx = Lxxt + Fxt.T @ (Vxx @ Fxt) 
            Qux = Luxt + Fut.T @ (Vxx @ Fxt)
            Quu = Luut + Fut.T @ (Vxx @ Fut)
            
            # regularization
            Quu_eigenvals, Quu_eigenvecs = np.linalg.eig(Quu)
            Quu_eigenvals[Quu_eigenvals < 0] = 0.0
            Quu_eigenvals += self.lamb
            Quu_inv = Quu_eigenvecs @ (np.diag(1.0/Quu_eigenvals) @ Quu_eigenvecs.T)

            k[t] = -Quu_inv @ Qu
            K[t] = -Quu_inv @ Qux

            Vx = Qx - K[t].T @ (Quu @ k[t])
            Vxx = Qxx - K[t].T @ (Quu @ K[t])
        
        return K, k

    def ilqr(self, x_init, U_init):

        trajectory = self.initial_rollout(x_init, U_init)
        #print(trajectory)
        control_sequence = U_init 
        trajectory_cost = self.get_trajectory_cost(trajectory, control_sequence)
       
        iterate = True
        epoch = 0

        while(epoch < self.max_epoch):
            epoch += 1

            K, k = self.backward_pass(trajectory, control_sequence)
            new_trajectory, new_control_sequence = self.forward_rollout(trajectory, control_sequence, K, k)
            #print(new_trajectory)
            new_trajectory_cost = self.get_trajectory_cost(new_trajectory, new_control_sequence)

            if(new_trajectory_cost < trajectory_cost):
                self.lamb /= self.lamb_factor

                if(abs((new_trajectory_cost - trajectory_cost) / trajectory_cost) < self.convergence_threshold):
                    break
            else:
                self.lamb *= self.lamb_factor
                if self.lamb > self.max_lamb:
                    break

            trajectory = new_trajectory
            control_sequence = new_control_sequence
            trajectory_cost = new_trajectory_cost
            
            sys.stdout.write("|Epoch: | Trajectory Cost: " + str(trajectory_cost) + " |\n")
        
        return trajectory, control_sequence   


