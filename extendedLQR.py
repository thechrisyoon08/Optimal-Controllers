"""
Implementation of Extended LQR:
"Extended LQR: Locally-Optimal Feedback Control for Systems with Non-Linear Dynamics and Non-Quadratic Cost,"
Jur van Den berg, 2014
"""

from autograd import grad, jacobian, hessian
import autograd.numpy as np

class ExtendedLQR():

    def __init__(self, system, cost, T, desired_min_total_cost):
        """
        params:
          * system: system from base class system.py
          * cost: system cost from base class cost.py
          * T: trajectory length
          * desired_min_total_cost: desired average total-cost (i.e. cost-to-go + cost-to-come) 
        """

        self.system = system
        self.cost = cost

        self.inverse_policy = None
        self.policy = [0]*T

        self.initial_backward = True
        self.initial_forward = True

        self.S = None
        self.Sbar = None
        self.sv = None
        self.svbar = None
        self.s = None

        self.x = None
        self.x_hat = None
        self.u = None
        self.u_hat = None
        self.T = T
        self.A = jacobian(self.system.dynamics, 0)
        self.B = jacobian(self.system.dynamics, 1)
        self.Clx = grad(self.cost.terminal_cost, 0)    
        self.Clxx = hessian(self.cost.terminal_cost, 0)         

    
        self.min_total_cost = desired_min_total_cost
        self.avg_trajectory_cost = 99999 
    
    def newTrajectory(self, newT):
        self.T = newT
        self.initial_backward = True
        self.initial_forward = True

    def calculate_total_cost(self, x, t):

        cost_to_go = (1/2) * x.T @ self.S[t] @ x + x.T @ self.sv[t]
        cost_to_come = (1/2) * x.T @ self.Sbar[t] @ x + x.T @ self.svbar[t] 

        total_cost = cost_to_go + cost_to_come
        return total_cost

    def average_cost_in_trajectory(self, X, U):
        ret = 0
        for t in range(self.T):
            ret += self.calculate_total_cost(X[t], t)
        ret = ret / self.T
        return ret
        
    def quadratize_cost(self, t):
        """
        params: 
          * x: state at time t
          * u: control at time t

        returns:
        Pt, Qt, Rt, qt, rt matrices
        """
        x = self.x_hat[t]
        u = self.u_hat[t]
        xu = np.concatenate((x, u))
        cost = self.cost.lagrangian
        def cost1(a):
            return cost(a[:len(x)],a[len(x):])
        #print(cost1(xu))
        QtPtPtRt = hessian(cost1)(xu)
        qtrt = grad(cost1)(xu)
        #print(QtPtPtRt)

        QtPtPtRt1 = np.split(QtPtPtRt,indices_or_sections=[len(x)], axis=1)
        QPR2 = list(map(lambda a:np.split(a, indices_or_sections=[len(x)]), QtPtPtRt1))

        return QPR2[0][1], QPR2[0][0], QPR2[1][1], qtrt[:len(x)], qtrt[len(x):]

    def inverse_dynamics(self, x, u):

        Abar_t = np.linalg.pinv(self.A(x,u))
        Bbar_t = -1 * Abar_t @ self.B(x,u)
        cbar_t = x - Abar_t.T @ x - np.reshape((np.array([Bbar_t]).T @ np.array([u])).T,-1,1)
        return Abar_t.T @ x + np.reshape((np.array([Bbar_t]).T @ np.array([u])).T,-1,1) + cbar_t


    def backward_pass(self):

        for t in range(self.T - 2, -1, -1):
            
            self.x_hat[t+1] = np.linalg.pinv(self.S[t+1] + self.Sbar[t+1]) @ (self.sv[t+1] + self.svbar[t+1])
            self.u_hat[t] = self.inverse_policy[t+1]    
            self.x_hat[t] = self.inverse_dynamics(self.x_hat[t+1], self.u_hat[t])

            # linearize dynamics about x_hat[t] and u_hat[t]
            A_t = self.A(self.x_hat[t], self.u_hat[t])
            B_t = self.B(self.x_hat[t], self.u_hat[t])
            c_t = self.x_hat[t+1] - A_t.T @ self.x_hat[t] - np.squeeze(np.array([B_t]).T @ np.array([self.u_hat[t]]))

            Pt, Qt, Rt, qt, rt = self.quadratize_cost(t)

            # compute St, sv_t, policy_t
            C_t = Pt + B_t.T @ self.S[t+1] @ A_t
            D_t = Qt + A_t.T @ self.S[t+1] @ A_t
            E_t = Rt + B_t.T @ self.S[t+1] @ B_t
            d_t = qt + A_t.T @ self.sv[t+1] + np.squeeze(A_t.T @ self.S[t+1] @ c_t)
            e_t = rt + B_t.T @ self.sv[t+1] + B_t.T @ self.S[t+1] @ c_t
            L_t = -1 * np.linalg.pinv(E_t) @ C_t
            l_t = -1 * np.linalg.pinv(E_t) @ e_t

            self.S[t] = D_t - C_t.T @ np.linalg.pinv(E_t) @ C_t
            self.sv[t] = d_t - C_t.T @ np.linalg.pinv(E_t) @ e_t

            #policy at time t given x_t, that is 
            self.policy[t] = L_t @ self.x_hat[t] + l_t 

        self.policy[self.T-1] = np.array([self.policy[self.T-1] + 0.00001])
    def forward_pass(self):
        
        for t in range(0, self.T-2, 1):
            if(t != 0):
                self.x_hat[t] = -1 * np.linalg.pinv(self.S[t] + self.Sbar[t]) @ (self.sv[t] + self.svbar[t])
            self.u_hat[t] = self.policy[t]
            self.x_hat[t+1] = self.system.dynamics(self.x_hat[t], self.u_hat[t])

            # solve eqn(8) to instead of taking dg/dx, dg/du as written in page 7
            Abar_t = np.linalg.pinv(self.A(self.x_hat[t], self.u_hat[t]))
            Bbar_t = -1 * Abar_t @ self.B(self.x_hat[t], self.u_hat[t])
            cbar_t = self.x_hat[t] - Abar_t @ self.x_hat[t+1] - Bbar_t @ self.u_hat[t]
            Pt, Qt, Rt, qt, rt = self.quadratize_cost(t)

            # compute Sbar[t], svbar[t] 
            Cbar_t = Bbar_t.T @ (self.Sbar[t] + Qt) @ Abar_t + Pt @ Abar_t
            Dbar_t = Abar_t.T @ (self.Sbar[t] + Qt) @ Abar_t
            Ebar_t = Bbar_t.T @ (self.Sbar[t] + Qt) @ Bbar_t + Rt @ Pt @ Bbar_t + Bbar_t.T @ Pt.T
            dbar_t = Abar_t.T @ (self.svbar[t] + qt) + Abar_t.T @ (self.Sbar[t] + Qt) @ cbar_t

            ebar_t = rt + Pt @ cbar_t + Bbar_t.T @ (self.svbar[t] + qt) + Bbar_t.T @ (self.Sbar[t] + Qt) @ cbar_t
            
            Ebar_t_inv = np.linalg.pinv(Ebar_t)
            Lbar_t = -1 * Ebar_t_inv @ Cbar_t
            lbar_t = -1 * Ebar_t_inv @ ebar_t
            self.inverse_policy[t] = Lbar_t @ self.x_hat[t+1] + lbar_t # double check the x_hat[t+1] tho

            self.Sbar[t+1] = Dbar_t - Cbar_t.T @ Ebar_t_inv @ Cbar_t
            self.svbar[t+1] = dbar_t - Cbar_t.T @ Ebar_t_inv @ ebar_t
            self.x_hat[-1] = -np.linalg.pinv(self.S[-1] + self.Sbar[-1]) @ (self.sv[-1] + self.svbar[-1])
    
    def extendedLQR(self, x_init):
        #repeat backward -> forward until convergence. eLQR is convergd when minimal-total-cost states (x_hat) no longer changes
        if(self.initial_backward):
            self.initial_backward = False
            self.x_hat = [None] * self.T
            self.x_hat[0] = x_init
            self.x_hat[-1] = x_init
            self.u_hat = [None] * self.T
            self.S = [np.zeros((self.system.state_dim, self.system.state_dim))] * self.T    # S = (n x n)
            self.S[-1] = self.Clxx(self.x_hat[-1])                 
            self.sv = [np.zeros(self.system.state_dim)] * self.T   # sv = (1 x n)
            self.sv[-1] = self.Clx(self.x_hat[-1]) - self.S[-1] @ self.x_hat[-1]  
            self.Sbar = [np.zeros((self.system.state_dim, self.system.state_dim))] * self.T
            self.svbar = [np.zeros(self.system.state_dim)] * self.T
            self.inverse_policy = [np.zeros(self.system.control_dim)] * self.T

        if(self.initial_forward):
            self.Sbar[0] = np.zeros((self.system.state_dim, self.system.state_dim))
            self.svbar[0] = np.zeros(self.system.state_dim)
            self.initial_forward = False
        
        while(self.avg_trajectory_cost > self.min_total_cost):
            self.backward_pass()
            self.forward_pass()
            self.avg_trajectory_cost = self.average_cost_in_trajectory(self.x_hat, self.u_hat)
        
        return self.x_hat, self.u_hat




        
        


