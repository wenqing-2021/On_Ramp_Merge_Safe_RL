# Model Predictive Control
# Author: Yuansj
# Time : 2022/02/24
import numpy as np
import time
import math
from cvxopt import matrix, solvers

class MPCSOLVE:
    """
    MPC solver used for merge scenario. Highway-env config "ContinuousAction" action
    """

    def __init__(self, rollout_space:int):
        self.N = rollout_space  # rollout space
        self.dt = 0.1  # time step
        self.state_n = 4 # features
        self.control_n = 2 # control num
        self.Length = 5  # m
        self.max_a = 5 # m/s^2
        self.max_steering_angle = np.pi / 180 * 5 # rad

        self.A_cell = None
        self.B_cell = None
        self.Esk = None

    def compute_matrix(self, car_info, ref_info) -> None:
        '''
        this function used to update matirx at the beginning of each control period
        input: k step velocity, heading, steering angle
        return: matrix p, matrix q, matrix g, matrix h, matrix a, matrix b
        '''
        position_k = car_info['position']
        velocity_k = car_info['velocity']
        steering_angle = car_info['steering_angle']
        heading_k = car_info['heading']
        E_s_k = np.mat([[position_k[0] - ref_info['p_ref'][0]],
                        [position_k[1] - ref_info['p_ref'][1]],
                        [velocity_k - ref_info['v_ref']],
                        [heading_k - ref_info['h_ref']]])
        
        v_k = velocity_k  # k step velocity
        beta = np.arctan(1 / 2 * np.tan(steering_angle))

        A = np.mat([[0, 0, np.cos(heading_k + beta), -v_k * np.sin(heading_k + beta)],
                    [0, 0, np.sin(heading_k + beta), v_k * np.cos(heading_k + beta)],
                    [0, 0, 0, 0],
                    [0, 0, np.sin(beta) / (self.Length / 2), 0]])

        B = np.mat([[0, -v_k * np.sin(heading_k + beta) * (2 / (1 + 3 * math.pow(np.cos(steering_angle), 2)))],
                    [0, v_k * np.cos(heading_k + beta) * (2 / (1 + 3 * math.pow(np.cos(steering_angle), 2)))],
                    [1, 0],
                    [0, v_k * np.cos(beta) * 2 / (self.Length / 2 * (1 + 3 * math.pow(np.cos(steering_angle), 2)))]])

        I = np.mat(np.eye(4))

        Akt = A * self.dt + I
        Bkt = B * self.dt

        # compute P and Q matrix
        current_A = Akt
        A_cell = current_A
        for i in range(self.N - 1):
            current_A = np.dot(current_A, Akt)
            A_cell = np.vstack((A_cell, current_A))

        B_cell_column = Bkt
        for j in range(self.N):
            current_b = Bkt
            if j == 0:
                for k in range(self.N - 1):
                    current_b = np.dot(Akt, current_b)
                    B_cell_column = np.vstack((B_cell_column , current_b))
                B_cell = B_cell_column
            else:
                B_cell_column = np.zeros([4*j,2])
                for k in range(self.N - j):
                    B_cell_column = np.vstack((B_cell_column , current_b))
                    current_b = np.dot(Akt, current_b)
                B_cell = np.hstack((B_cell, B_cell_column))
        
        A_cell = np.mat(A_cell)
        B_cell = np.mat(B_cell)
        Q = np.mat(np.eye(self.N * self.state_n))
        R = np.mat(np.eye(self.N * self.control_n))

        P = 2*B_cell.transpose()*Q*B_cell + R
        Q = 2*(E_s_k.transpose()*A_cell.transpose()*Q*B_cell).transpose()

        # compute constraint matrix
        g_matrix_1 = np.eye(self.N * self.control_n)
        h_matrix_1 = np.zeros([self.N * self.control_n, 1])
        for i in range(len(h_matrix_1)):
            if i % 2 == 0:
                h_matrix_1[i][0] = self.max_a
            elif i % 2 == 1:
                h_matrix_1[i][0] = self.max_steering_angle
        
        h_matrix_2 = h_matrix_1
        g_matrix_2 = - g_matrix_1

        G = np.mat(np.vstack((g_matrix_1, g_matrix_2)))
        H = np.mat(np.vstack((h_matrix_1, h_matrix_2)))

        self.A_cell = A_cell
        self.B_cell = B_cell
        self.Esk = E_s_k

        return P, Q, G, H

    def solve(self, P, Q, G, H, A=[], B=[]):
        '''
        solve the QP
        input: 
        P, Q is the matrix of objective function
        G, H is the inequality constraint equation coefficient
        A, B is the equality constraint equation coefficient
        return: optimal control u
        '''
        p = matrix(P)
        q = matrix(Q)
        g = matrix(G)
        h = matrix(H)
        a = matrix(A)
        b = matrix(B)

        settings = dict(show_progress=False)
        sol = solvers.qp(p,q,g,h,options=settings)
        result = sol['x']

        return result

