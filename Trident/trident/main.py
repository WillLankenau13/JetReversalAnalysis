import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from numba import jit
# import os
# import random
# import psycopg2
import json

# jit makes code run faster
@jit(nopython=True)
def simulate_numba(
    num_steps,
    k_e_square,
    W_e,
    L_e_plus,
    W_plus,
    L_plus_e,
    eta_batch,
    dt,
    epsilon,
    r_m,
    k_plus_square,
    k,
    phi_e,
    phi_plus,
    U
):
    
    #we define zeroed arrays to store values at each timestep
    phi_e_history = np.zeros((num_steps, 2))
    phi_plus_history = np.zeros((num_steps, 2))
    U_history = np.zeros(num_steps)
    R_vals = np.zeros(num_steps)

    k_e_psi_e_vals = np.zeros(num_steps)
    k_e_b_e_vals = np.zeros(num_steps)
    k_e_psi_plus_vals = np.zeros(num_steps)
    k_e_b_plus_vals = np.zeros(num_steps)
    heat_flux_psi_e_b_e_vals = np.zeros(num_steps)
    heat_flux_psi_e_b_plus_vals = np.zeros(num_steps)
    b_e_psi_plus_vals = np.zeros(num_steps)
    b_e_b_plus_vals = np.zeros(num_steps)
    psi_plus_b_plus_vals = np.zeros(num_steps)

    jacobians = np.zeros((num_steps, 5, 5))

    switch_times = []

    for i in range(num_steps):
        eta = eta_batch[i] #etas are random numbers generated from N(0,1) (if we are in random and not deterministic mode)
        xi = np.array([2 * np.sqrt(2) * eta[0] / np.sqrt(k_e_square), 0.0]) #stochastic forcing vector
        phi_e_dot = np.zeros(2)
        phi_plus_dot = np.zeros(2)

        jacobian = np.zeros((5, 5))

        #equation 2.10
        #why are we taking the sqrt of dt? something to do with Euler? but why inverse sqrt?
        #claude says its because we multiply by dt later, but we want the factor to be sqrt(dt) (so that variance is dt for our random number from a N(0,1)). So, we divide by sqrt(dt) here
        phi_e_dot[0] = W_e[0, 0] * phi_e[0] + W_e[0, 1] * phi_e[1] + U * (
            L_e_plus[0, 0] * phi_plus[0] + L_e_plus[0, 1] * phi_plus[1]) + (np.sqrt(epsilon) * xi[0]) / np.sqrt(dt)
        phi_e_dot[1] = W_e[1, 0] * phi_e[0] + W_e[1, 1] * phi_e[1] + U * (
            L_e_plus[1, 0] * phi_plus[0] + L_e_plus[1, 1] * phi_plus[1]) + (np.sqrt(epsilon) * xi[1]) / np.sqrt(dt)

        #no stochastic forcing for phi_plus_dot? why?
        #is it because we are only adding randomness in one direction?
        #likely because stochastic forcing vector is only nonzero in one term
        phi_plus_dot[0] = W_plus[0, 0] * phi_plus[0] + W_plus[0, 1] * phi_plus[1] + U * (
            L_plus_e[0, 0] * phi_e[0] + L_plus_e[0, 1] * phi_e[1])
        phi_plus_dot[1] = W_plus[1, 0] * phi_plus[0] + W_plus[1, 1] * phi_plus[1] + U * (
            L_plus_e[1, 0] * phi_e[0] + L_plus_e[1, 1] * phi_e[1])

        #putting parts of equation 2.10 into the jacobian? why?
        #these are the partial derivatives of equation 2.10 wrt phi_e[0] (psi_e), phi_e[1] (b_e), phi_plus[0] (psi_plus), phi_plus[1] (b_plus), and U
        jacobian[0, 0] = W_e[0, 0]
        jacobian[0, 1] = W_e[0, 1]
        jacobian[0, 2] = U * L_e_plus[0, 0]
        jacobian[0, 3] = U * L_e_plus[0, 1]
        jacobian[0, 4] = L_e_plus[0, 0] * phi_plus[0] + L_e_plus[0, 1] * phi_plus[1]

        jacobian[1, 0] = W_e[1, 0]
        jacobian[1, 1] = W_e[1, 1]
        jacobian[1, 2] = U * L_e_plus[1, 0]
        jacobian[1, 3] = U * L_e_plus[1, 1]
        jacobian[1, 4] = L_e_plus[1, 0] * phi_plus[0] + L_e_plus[1, 1] * phi_plus[1]

        jacobian[2, 0] = U * L_plus_e[0, 0]
        jacobian[2, 1] = U * L_plus_e[0, 1]
        jacobian[2, 2] = W_plus[0, 0]
        jacobian[2, 3] = W_plus[0, 1]
        jacobian[2, 4] = L_plus_e[0, 0] * phi_e[0] + L_plus_e[0, 1] * phi_e[1]

        jacobian[3, 0] = U * L_plus_e[1, 0]
        jacobian[3, 1] = U * L_plus_e[1, 1]
        jacobian[3, 2] = W_plus[1, 0]
        jacobian[3, 3] = W_plus[1, 1]
        jacobian[3, 4] = L_plus_e[1, 0] * phi_e[0] + L_plus_e[1, 1] * phi_e[1]

        #adjusting values based on derivative and timestep; Euler
        phi_e[0] += phi_e_dot[0] * dt
        phi_e[1] += phi_e_dot[1] * dt
        phi_plus[0] += phi_plus_dot[0] * dt
        phi_plus[1] += phi_plus_dot[1] * dt

        #defining R and U_dot, equations 2.12 and 2.13
        R = 0.25 * k * (k_plus_square - k_e_square) * phi_e[0] * phi_plus[0]
        U_dot = R - r_m * U

        #putting more garbage into the jacobian
        jacobian[4, 0] = 0.25 * k * (k_plus_square - k_e_square) * phi_plus[0]
        jacobian[4, 1] = 0
        jacobian[4, 2] = 0.25 * k * (k_plus_square - k_e_square) * phi_e[0]
        jacobian[4, 3] = 0
        jacobian[4, 4] = -r_m

        #more euler updating
        U += U_dot * dt

        #store values in history arrays
        phi_e_history[i, 0] = phi_e[0]
        phi_e_history[i, 1] = phi_e[1]
        phi_plus_history[i, 0] = phi_plus[0]
        phi_plus_history[i, 1] = phi_plus[1]

        #temp vars
        psi_e = phi_e[0]
        b_e = phi_e[1]
        psi_plus = phi_plus[0]
        b_plus = phi_plus[1]

        #store more values in history arrays
        U_history[i] = U
        R_vals[i] = R

        #store even more values in history arrays
        #this is the covariance matrix
        k_e_psi_e_vals[i] = psi_e*psi_e
        k_e_b_e_vals[i] = b_e*b_e
        k_e_psi_plus_vals[i] = psi_plus*psi_plus
        k_e_b_plus_vals[i] = b_plus*b_plus
        heat_flux_psi_e_b_e_vals[i] = psi_e*b_e
        heat_flux_psi_e_b_plus_vals[i] = psi_e*b_plus
        b_e_psi_plus_vals[i] = b_e*psi_plus
        b_e_b_plus_vals[i] = b_e*b_plus
        psi_plus_b_plus_vals[i] = psi_plus*b_plus

        jacobians[i] = jacobian

        #check if there was a reversal event
        #pending further conditions
        if i > 0:
            if U_history[i - 1] > 0 and U_history[i] < 0:
                switch_times.append(i)
    # return phi_e_history, phi_plus_history, U_history, R_vals, k_e_psi_e_vals, k_e_b_e_vals, k_e_psi_plus_vals, k_e_b_plus_vals, heat_flux_psi_e_b_e_vals, heat_flux_psi_e_b_plus_vals, b_e_psi_plus_vals, b_e_b_plus_vals, psi_plus_b_plus_vals, switch_times
    return (
        phi_e_history,
        phi_plus_history,
        U_history,
        R_vals,
        k_e_psi_e_vals,
        k_e_b_e_vals,
        k_e_psi_plus_vals,
        k_e_b_plus_vals,
        heat_flux_psi_e_b_e_vals,
        heat_flux_psi_e_b_plus_vals,
        b_e_psi_plus_vals,
        b_e_b_plus_vals,
        psi_plus_b_plus_vals,
        jacobians,
        switch_times
    )

class Simulation:
    def __init__(self, epsilon, N_0_squared, r_m, k, m, m_u, dt, total_time, randomness):
        self.epsilon = epsilon
        self.N_0_squared = N_0_squared
        self.r_m = r_m
        self.k = k
        self.m = m
        self.m_u = m_u
        self.dt = dt
        self.total_time = total_time
        self.num_steps = int(total_time / dt) #calculate timesteps
        self.k_e_square = k**2 + m**2 #define initial k_e square
        self.k_plus_square = k**2 + (m + m_u)**2 #define initial k_plus square

        #define W_e and W_plus, equation 2.14
        self.W_e = np.array([[-1, (k / self.k_e_square)], [-k * N_0_squared, -1]])
        self.W_plus = np.array([[-1, -k / self.k_plus_square], [k * N_0_squared, -1]])
        
        #define L_e_plus and L_plus_e, equation 2.15
        self.L_e_plus = np.array([[(-k / (2 * self.k_e_square)) * (self.k_plus_square - m_u**2), 0],
                                  [0, k / 2]])
        self.L_plus_e = np.array([[(-k / (2 * self.k_plus_square)) * (m_u**2 - self.k_e_square), 0],
                                  [0, -k / 2]])

        #initialize history arrays
        #this is the covariance matrix
        self.phi_e_history = np.zeros((self.num_steps, 2))
        self.phi_plus_history = np.zeros((self.num_steps, 2))
        self.U_history = np.zeros(self.num_steps)
        self.R_vals = np.zeros(self.num_steps)
        self.k_e_psi_e_vals = np.zeros(self.num_steps)
        self.k_e_b_e_vals = np.zeros(self.num_steps)
        self.k_e_psi_plus_vals = np.zeros(self.num_steps)
        self.k_e_b_plus_vals = np.zeros(self.num_steps)
        self.heat_flux_psi_e_b_e_vals = np.zeros(self.num_steps)
        self.heat_flux_psi_e_b_plus_vals = np.zeros(self.num_steps)
        self.b_e_psi_plus_vals = np.zeros(self.num_steps)
        self.b_e_b_plus_vals = np.zeros(self.num_steps)
        self.psi_plus_b_plus_vals = np.zeros(self.num_steps)

        #more initialization
        self.randomness = randomness
        self.jacobians = np.zeros((self.num_steps, 5, 5))
        self.switch_times = None
        
        #define dictionary for reversals
        self.reversals = {
            "timesteps": [],
            "data": []
        }

        self.eta_batch = self.generate_eta_batch()

    #function to generate eta values
    #random numbers from a normal, one for each timestep
    def generate_eta_batch(self):
        if(self.randomness):
            return np.random.normal(0, 1, size=(self.num_steps, 1))
        return np.zeros((self.num_steps, 1))
    
    #function to get the maximum eigenvalue from the jacobians at each timestep
    #measuring the local stability of the system
    def get_dominant_eigenvalues(self):
        dominant_eigenvals = np.zeros(self.num_steps)
        for i in range(self.num_steps):
            vals, _ = np.linalg.eig(self.jacobians[i])
            dominant_eigenvals[i] = np.max(vals.real)

        return dominant_eigenvals

    # def simulate(self):
    #     self.phi_e_history, self.phi_plus_history, self.U_history, self.R_vals, self.k_e_psi_e_vals, self.k_e_b_e_vals, self.k_e_psi_plus_vals, self.k_e_b_plus_vals, self.heat_flux_psi_e_b_e_vals, self.heat_flux_psi_e_b_plus_vals, self.b_e_psi_plus_vals, self.b_e_b_plus_vals, self.psi_plus_b_plus_vals, self.switch_times = simulate_numba(
    #         self.num_steps, self.k_e_square, self.W_e, self.L_e_plus, self.W_plus, self.L_plus_e, self.eta_batch, self.dt, self.epsilon, self.r_m, self.k_plus_square, self.k)
    #     return len(self.switch_times), [t * self.dt for t in self.switch_times]

    def simulate(self, phi_e, phi_plus, U):
        (
            self.phi_e_history,
            self.phi_plus_history,
            self.U_history,
            self.R_vals,
            self.k_e_psi_e_vals,
            self.k_e_b_e_vals,
            self.k_e_psi_plus_vals,
            self.k_e_b_plus_vals,
            self.heat_flux_psi_e_b_e_vals,
            self.heat_flux_psi_e_b_plus_vals,
            self.b_e_psi_plus_vals,
            self.b_e_b_plus_vals,
            self.psi_plus_b_plus_vals,
            self.jacobians,
            self.switch_times
        ) = simulate_numba(
            self.num_steps,
            self.k_e_square,
            self.W_e,
            self.L_e_plus,
            self.W_plus,
            self.L_plus_e,
            self.eta_batch,
            self.dt,
            self.epsilon,
            self.r_m,
            self.k_plus_square,
            self.k,
            phi_e,
            phi_plus,
            U
        )

        # return len(self.switch_times), [t * self.dt for t in self.switch_times]

    #put it in json format
    def get_json_simulation_data(self, target_features):
        # print("    - Getting data in json format...")
        data_dict = {
            "eps": f"{self.epsilon:.5f}",
            "n_0_squared": f"{self.N_0_squared:.5f}",
            "psi_e": json.dumps(self.phi_e_history[:, 0].tolist()),
            "b_e": json.dumps(self.phi_e_history[:, 1].tolist()),
            "psi_plus": json.dumps(self.phi_plus_history[:, 0].tolist()),
            "b_plus": json.dumps(self.phi_plus_history[:, 1].tolist()),
            "u_list": json.dumps(self.U_history.tolist()),
            "r_list": json.dumps(self.R_vals.tolist()),
            "k_e_psi_e_list": json.dumps(self.k_e_psi_e_vals.tolist()),
            "k_e_b_e_list": json.dumps(self.k_e_b_e_vals.tolist()),
            "k_e_psi_plus_list": json.dumps(self.k_e_psi_plus_vals.tolist()),
            "k_e_b_plus_list": json.dumps(self.k_e_b_plus_vals.tolist()),
            "heat_flux_psi_e_b_e_list": json.dumps(self.heat_flux_psi_e_b_e_vals.tolist()),
            "heat_flux_psi_e_b_plus_list": json.dumps(self.heat_flux_psi_e_b_plus_vals.tolist()),
            "b_e_psi_plus_list": json.dumps(self.b_e_psi_plus_vals.tolist()),
            "b_e_b_plus_list": json.dumps(self.b_e_b_plus_vals.tolist()),
            "psi_plus_b_plus_list": json.dumps(self.psi_plus_b_plus_vals.tolist()),
            "eta_list": json.dumps(self.eta_batch.tolist()),
            "dom_eigenvals": json.dumps(self.get_dominant_eigenvalues().tolist())
        }

        return [{feature: data_dict[feature] for feature in target_features if feature in data_dict}]


    #reversal data to json format
    def get_json_reversal_data(self, target_features):
        json_data = []

        for reversal in self.reversals["data"]:
            json_reversal = {
                "eps": f"{self.epsilon:.5f}",
                "n_0_squared": f"{self.N_0_squared:.5f}"
            }
            for feature, vals in reversal.items():
                if(feature in target_features):
                    json_reversal[feature] = json.dumps(vals.tolist())
            json_data.append(json_reversal)

        return json_data


    # def extract_reversal_data(self, window_size=5000):
    #     reversal_data = {}
    #     step_units = window_size
    #     last_reversal_index = -step_units  

    #     for switch_time in self.switch_times:
    #         index = switch_time
    #         if index - last_reversal_index < step_units:
    #             continue  
            

    #         if index >= step_units and index + step_units < self.num_steps:
    #             pre_reversal_positive = np.all(self.U_history[index - step_units:index] > 0)
    #             post_reversal_negative = np.all(self.U_history[index:index + step_units] < 0)
                
    #             if not pre_reversal_positive or not post_reversal_negative:
    #                 continue 
                
    #             reversal_data[f"reversal_at_{switch_time}"] = {
    #                 'phi_e': self.phi_e_history[index - step_units:index + step_units],
    #                 'phi_plus': self.phi_plus_history[index - step_units:index + step_units],
    #                 'U': self.U_history[index - step_units:index + step_units],
    #                 'R': self.R_vals[index - step_units:index + step_units],
    #                 'k_e_psi_e':self.k_e_psi_e_vals[index - step_units:index + step_units],
    #                 'k_e_b_e':self.k_e_b_e_vals[index - step_units:index + step_units],
    #                 'k_e_psi_plus':self.k_e_psi_plus_vals[index - step_units:index + step_units],
    #                 'k_e_b_plus':self.k_e_b_plus_vals[index - step_units:index + step_units],
    #                 'heat_flux_psi_e_b_e':self.heat_flux_psi_e_b_e_vals[index - step_units:index + step_units],
    #                 'heat_flux_psi_e_b_plus':self.heat_flux_psi_e_b_plus_vals[index - step_units:index + step_units],
    #                 'b_e_psi_plus':self.b_e_psi_plus_vals[index - step_units:index + step_units],
    #                 'b_e_b_plus':self.b_e_b_plus_vals[index - step_units:index + step_units],
    #                 'psi_plus_b_plus':self.psi_plus_b_plus_vals[index - step_units:index + step_units],
    #                 'eta': self.eta_batch[index - step_units:index + step_units]
    #             }
    #         last_reversal_index = index
    #     return reversal_data

    #algorithm to get only good reversal data
    #can't have another reversal in the 5000 timesteps before
    #store the data in reversals dictionary
    #we take only 1/10 timesteps
    def extract_compressed_reversal_data(self, window_size = 5000):
        step_units = window_size
        last_reversal_index = -step_units

        for switch_time in self.switch_times:
            index = switch_time
            if index - last_reversal_index < step_units:
                continue  
            
            if index >= step_units and index + step_units < self.num_steps:
                pre_reversal_positive = np.all(self.U_history[index - step_units:index] > 0)
                post_reversal_negative = np.all(self.U_history[index:index + step_units] < 0)

                if not pre_reversal_positive or not post_reversal_negative:
                    continue
                
                self.reversals["timesteps"].append(switch_time)
                self.reversals["data"].append(
                    {
                        "psi_e": self.phi_e_history[index - step_units:index + step_units][:, 0][9::10],
                        "b_e": self.phi_e_history[index - step_units:index + step_units][:, 1][9::10],
                        "psi_plus": self.phi_plus_history[index - step_units:index + step_units][:, 0][9::10],
                        "b_plus": self.phi_plus_history[index - step_units:index + step_units][:, 1][9::10],
                        "u_list": self.U_history[index - step_units:index + step_units][9::10],
                        "r_list": self.R_vals[index - step_units:index + step_units][9::10],
                        "k_e_psi_e_list": self.k_e_psi_e_vals[index - step_units:index + step_units][9::10],
                        "k_e_b_e_list": self.k_e_b_e_vals[index - step_units:index + step_units][9::10],
                        "k_e_psi_plus_list": self.k_e_psi_plus_vals[index - step_units:index + step_units][9::10],
                        "k_e_b_plus_list": self.k_e_b_plus_vals[index - step_units:index + step_units][9::10],
                        "heat_flux_psi_e_b_e_list": self.heat_flux_psi_e_b_e_vals[index - step_units:index + step_units][9::10],
                        "heat_flux_psi_e_b_plus_list": self.heat_flux_psi_e_b_plus_vals[index - step_units:index + step_units][9::10],
                        "b_e_psi_plus_list": self.b_e_psi_plus_vals[index - step_units:index + step_units][9::10],
                        "b_e_b_plus_list": self.b_e_b_plus_vals[index - step_units:index + step_units][9::10],
                        "psi_plus_b_plus_list": self.psi_plus_b_plus_vals[index - step_units:index + step_units][9::10],
                        "eta_list": self.eta_batch[index - step_units:index + step_units][9::10],
                        "dom_eigenvals": self.get_dominant_eigenvalues()[index - step_units:index + step_units][9::10]
                    }
                )
            last_reversal_index = index
    
    # def compress(self):
    #     self.phi_e_history = self.phi_e_history[9::10]
    #     self.phi_plus_history = self.phi_plus_history[9::10]
    #     self.U_history = self.U_history[9::10]
    #     self.R_vals = self.R_vals[9::10]
    #     self.k_e_psi_e_vals = self.k_e_psi_e_vals[9::10]
    #     self.k_e_b_e_vals = self.k_e_b_e_vals[9::10]
    #     self.k_e_psi_plus_vals = self.k_e_psi_plus_vals[9::10]
    #     self.k_e_b_plus_vals = self.k_e_b_plus_vals[9::10]
    #     self.heat_flux_psi_e_b_e_vals = self.heat_flux_psi_e_b_e_vals[9::10]
    #     self.heat_flux_psi_e_b_plus_vals = self.heat_flux_psi_e_b_plus_vals[9::10]
    #     self.b_e_psi_plus_vals = self.b_e_psi_plus_vals[9::10]
    #     self.b_e_b_plus_vals = self.b_e_b_plus_vals[9::10]
    #     self.psi_plus_b_plus_vals = self.psi_plus_b_plus_vals[9::10]
    #     self.eta_batch = self.eta_batch[9::10]

        

# def average_arrays(*arrays):
#     if not arrays:
#         raise ValueError("No arrays provided for averaging.")
    
#     np_arrays = [np.array(arr) for arr in arrays]
#     array_lengths = [len(arr) for arr in np_arrays]

#     if len(set(array_lengths)) != 1:
#         raise ValueError("All input arrays must have the same length.")
    
#     average_array = np.mean(np_arrays, axis=0)
    
#     return average_array

#This might still be useful to encapsulate the current aggregator into a function.

# def reversal_data_aggregator(window_size=5000):
#     phi_e_list = []
#     phi_plus_list = []
#     U_list = []
#     R_list = []
#     k_e_psi_e_list = []
#     k_e_b_e_list = []
#     k_e_psi_plus_list = []
#     k_e_b_plus_list = []
#     heat_flux_psi_e_b_e_list = []
#     heat_flux_psi_e_b_plus_list = []
#     b_e_psi_plus_list = []
#     b_e_b_plus_list = []
#     psi_plus_b_plus_list = []
#     eta_list = []

#     for sim in simulations:
#         reversal_data = sim.extract_reversal_data(window_size)
#         for key in reversal_data:
#             phi_e_list.append(reversal_data[key]['phi_e'])
#             phi_plus_list.append(reversal_data[key]['phi_plus'])
#             U_list.append(reversal_data[key]['U'])
#             R_list.append(reversal_data[key]['R'])
#             k_e_psi_e_list.append(reversal_data[key]['k_e_psi_e'])
#             k_e_b_e_list.append(reversal_data[key]['k_e_b_e'])
#             k_e_psi_plus_list.append(reversal_data[key]['k_e_psi_plus'])
#             k_e_b_plus_list.append(reversal_data[key]['k_e_b_plus'])
#             heat_flux_psi_e_b_e_list.append(reversal_data[key]['heat_flux_psi_e_b_e'])
#             heat_flux_psi_e_b_plus_list.append(reversal_data[key]['heat_flux_psi_e_b_plus'])
#             b_e_psi_plus_list.append(reversal_data[key]['b_e_psi_plus'])
#             b_e_b_plus_list.append(reversal_data[key]['b_e_b_plus'])
#             psi_plus_b_plus_list.append(reversal_data[key]['psi_plus_b_plus'])
#             eta_list.append(reversal_data[key]['eta'])


# def save_simulation_data(simulations, output_file='all_simulation_data.csv'):
#     nested_data = {
#         'simulation_index': [],
#         'U_history': [],
#         'R_vals': [],
#         'k_e_psi_e_vals': [],
#         'k_e_b_e_vals': [],
#         'k_e_psi_plus_vals': [],
#         'k_e_b_plus_vals': [],
#         'heat_flux_psi_e_b_e_vals': [],
#         'heat_flux_psi_e_b_plus_vals': [],
#         'b_e_psi_plus_vals': [],
#         'b_e_b_plus_vals': [],
#         'psi_plus_b_plus_vals': [],
#         'eta_batch': []
#     }

#     for idx, sim in enumerate(simulations):
#         nested_data['simulation_index'].append(idx)
#         nested_data['U_history'].append(sim.U_history.tolist())
#         nested_data['R_vals'].append(sim.R_vals.tolist())
#         nested_data['k_e_psi_e_vals'].append(sim.k_e_psi_e_vals.tolist())
#         nested_data['k_e_b_e_vals'].append(sim.k_e_b_e_vals.tolist())
#         nested_data['k_e_psi_plus_vals'].append(sim.k_e_psi_plus_vals.tolist())
#         nested_data['k_e_b_plus_vals'].append(sim.k_e_b_plus_vals.tolist())
#         nested_data['heat_flux_psi_e_b_e_vals'].append(sim.heat_flux_psi_e_b_e_vals.tolist())
#         nested_data['heat_flux_psi_e_b_plus_vals'].append(sim.heat_flux_psi_e_b_plus_vals.tolist())
#         nested_data['b_e_psi_plus_vals'].append(sim.b_e_psi_plus_vals.tolist())
#         nested_data['b_e_b_plus_vals'].append(sim.b_e_b_plus_vals.tolist())
#         nested_data['psi_plus_b_plus_vals'].append(sim.psi_plus_b_plus_vals.tolist())
#         nested_data['eta_batch'].append(sim.eta_batch.flatten().tolist())

#     df = pd.DataFrame(nested_data)
#     df.to_csv(output_file, index=False)
#     print(f"All simulation data saved to {output_file}")

# if __name__ == "__main__":
#     # simulations = []
#     compositeHalfList = []
#     epsilon = 0.12394270273516043
#     N_0_squared = 318.8640217310387
#     r_m = 0.1
#     k = 2 * np.pi * 6
#     m = 2 * np.pi * 3
#     m_u = 2 * np.pi * 7
#     dt = 0.001
#     total_time = 200

#     phi_e_list = []
#     phi_plus_list = []
#     U_list = []
#     R_list = []
#     k_e_psi_e_list = []
#     k_e_b_e_list = []
#     k_e_psi_plus_list = []
#     k_e_b_plus_list = []
#     heat_flux_psi_e_b_e_list = []
#     heat_flux_psi_e_b_plus_list = []
#     b_e_psi_plus_list = []
#     b_e_b_plus_list = []
#     psi_plus_b_plus_list = []
#     eta_list = []

#     for i in range(3800):
#         print(f"Running iteration {i}")
#         sim = Simulation(epsilon, N_0_squared, r_m, k, m, m_u, dt, total_time)
#         sim.simulate()
#         reversal_data = sim.extract_reversal_data(window_size=5000)
#         for key in reversal_data:
#             phi_e_list.append(reversal_data[key]['phi_e'][9::10])
#             phi_plus_list.append(reversal_data[key]['phi_plus'][9::10])
#             U_list.append(reversal_data[key]['U'][9::10])
#             R_list.append(reversal_data[key]['R'][9::10])
#             k_e_psi_e_list.append(reversal_data[key]['k_e_psi_e'][9::10])
#             k_e_b_e_list.append(reversal_data[key]['k_e_b_e'][9::10])
#             k_e_psi_plus_list.append(reversal_data[key]['k_e_psi_plus'][9::10])
#             k_e_b_plus_list.append(reversal_data[key]['k_e_b_plus'][9::10])
#             heat_flux_psi_e_b_e_list.append(reversal_data[key]['heat_flux_psi_e_b_e'][9::10])
#             heat_flux_psi_e_b_plus_list.append(reversal_data[key]['heat_flux_psi_e_b_plus'][9::10])
#             b_e_psi_plus_list.append(reversal_data[key]['b_e_psi_plus'][9::10])
#             b_e_b_plus_list.append(reversal_data[key]['b_e_b_plus'][9::10])
#             psi_plus_b_plus_list.append(reversal_data[key]['psi_plus_b_plus'][9::10])
#             eta_list.append(reversal_data[key]['eta'][9::10])
#         # del sim

#     conn = psycopg2.connect("dbname=simulations_data user=simulationuser password=simulations2024 host=localhost")
#     cur = conn.cursor()

#     # psi_e = [item[0] for item in phi_e_list]
#     # b_e = [item[1] for item in phi_e_list]
#     # psi_plus = [item[0] for item in phi_plus_list]
#     # b_plus = [item[1] for item in phi_plus_list]

#     for i in range(len(U_list)):

#         psi_e_json = json.dumps(phi_e_list[i][:, 0].tolist())
#         b_e_json = json.dumps(phi_e_list[i][:, 1].tolist())
#         psi_plus_json = json.dumps(phi_plus_list[i][:, 0].tolist())
#         b_plus_json = json.dumps(phi_plus_list[i][:, 1].tolist())
#         U_list_json = json.dumps(U_list[i].tolist())
#         R_list_json = json.dumps(R_list[i].tolist())
#         k_e_psi_e_list_json = json.dumps(k_e_psi_e_list[i].tolist())
#         k_e_b_e_list_json = json.dumps(k_e_b_e_list[i].tolist())
#         k_e_psi_plus_list_json = json.dumps(k_e_psi_plus_list[i].tolist())
#         k_e_b_plus_list_json = json.dumps(k_e_b_plus_list[i].tolist())
#         heat_flux_psi_e_b_e_list_json = json.dumps(heat_flux_psi_e_b_e_list[i].tolist())
#         heat_flux_psi_e_b_plus_list_json = json.dumps(heat_flux_psi_e_b_plus_list[i].tolist())
#         b_e_psi_plus_list_json = json.dumps(b_e_psi_plus_list[i].tolist())
#         b_e_b_plus_list_json = json.dumps(b_e_b_plus_list[i].tolist())
#         psi_plus_b_plus_list_json = json.dumps(psi_plus_b_plus_list[i].tolist())
#         eta_list_json = json.dumps(eta_list[i].tolist())

#     ##The variables in the for loop have weird names. They are not lists.

#         cur.execute("""
#         INSERT INTO composite_data (eps, n_0_squared, psi_e, b_e, psi_plus, b_plus, U_list, R_list, k_e_psi_e_list, 
#                                 k_e_b_e_list, k_e_psi_plus_list, k_e_b_plus_list, heat_flux_psi_e_b_e_list, 
#                                 heat_flux_psi_e_b_plus_list, b_e_psi_plus_list, b_e_b_plus_list, 
#                                 psi_plus_b_plus_list, eta_list) 
#         VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
#         """, (epsilon, N_0_squared, psi_e_json, b_e_json, psi_plus_json, b_plus_json, U_list_json, R_list_json, 
#             k_e_psi_e_list_json, k_e_b_e_list_json, k_e_psi_plus_list_json, k_e_b_plus_list_json, 
#             heat_flux_psi_e_b_e_list_json, heat_flux_psi_e_b_plus_list_json, b_e_psi_plus_list_json, 
#             b_e_b_plus_list_json, psi_plus_b_plus_list_json, eta_list_json))

#     conn.commit()
#     cur.close()
#     conn.close()
#     print("Inserted batch into database")

#     # plot_composite_analysis("totalPlot", epsilon, N_0_squared, phi_e_list, phi_plus_list, U_list, R_list, k_e_psi_e_list, k_e_b_e_list, k_e_psi_plus_list, k_e_b_plus_list, heat_flux_psi_e_b_e_list, heat_flux_psi_e_b_plus_list, b_e_psi_plus_list, b_e_b_plus_list, psi_plus_b_plus_list, eta_list, dt,2000)
#     # save_simulation_data(simulations)