import os
import sys
import pandas
import pickle
import argparse
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.io
from scipy.optimize import curve_fit, least_squares, minimize
import math
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
import quadprog
import cvxopt
from scipy import linalg as la
from scipy import special

from data_management.read_csv import *
from data_management.functions import *
from visualization.visualize_frame import VisualizationPlot

np.set_printoptions(suppress = True)

data_num = "01"
# path = "D:/learn/23SS/guided research/highd-dataset-v1.0/highD-dataset/Python/data/"
# print(os.getcwd())

################################################################# provided codes for reading data ################################################################
def create_args():
    parser = argparse.ArgumentParser(description="ParameterOptimizer")
    # --- Input paths ---
    parser.add_argument('--input_path', default="D:/learn/23SS/guided_research/highd-dataset-v1.0/data/"+data_num+"_tracks.csv", type=str,
                        help='CSV file of the tracks')
    parser.add_argument('--input_static_path', default="D:/learn/23SS/guided_research/highd-dataset-v1.0/data/"+data_num+"_tracksMeta.csv",
                        type=str,
                        help='Static meta data file for each track')
    parser.add_argument('--input_meta_path', default="D:/learn/23SS/guided_research/highd-dataset-v1.0/data/"+data_num+"_recordingMeta.csv",
                        type=str,
                        help='Static meta data file for the whole video')
    parser.add_argument('--pickle_path', default="D:/learn/23SS/guided_research/highd-dataset-v1.0/data/"+data_num+".pickle", type=str,
                        help='Converted pickle file that contains corresponding information of the "input_path" file')
    # --- Settings ---
    parser.add_argument('--visualize', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='True if you want to visualize the data.')
    parser.add_argument('--background_image', default="D:/learn/23SS/guided_research/highd-dataset-v1.0/data/"+data_num+"_highway.jpg", type=str,
                        help='Optional: you can specify the correlating background image.')

    # --- Visualization settings ---
    parser.add_argument('--plotBoundingBoxes', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Optional: decide whether to plot the bounding boxes or not.')
    parser.add_argument('--plotDirectionTriangle', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Optional: decide whether to plot the direction triangle or not.')
    parser.add_argument('--plotTextAnnotation', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Optional: decide whether to plot the text annotation or not.')
    parser.add_argument('--plotTrackingLines', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Optional: decide whether to plot the tracking lines or not.')
    parser.add_argument('--plotClass', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Optional: decide whether to show the class in the text annotation.')
    parser.add_argument('--plotVelocity', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Optional: decide whether to show the class in the text annotation.')
    parser.add_argument('--plotIDs', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Optional: decide whether to show the class in the text annotation.')

    # --- I/O settings ---
    parser.add_argument('--save_as_pickle', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Optional: you can save the tracks as pickle.')
    parsed_arguments = vars(parser.parse_args())
    return parsed_arguments

if __name__ == '__main__':
    created_arguments = create_args()
    print("Try to find the saved pickle file for better performance.")
    # Read the track csv and convert to useful format
    if os.path.exists(created_arguments["pickle_path"]):
        with open(created_arguments["pickle_path"], "rb") as fp:
            tracks = pickle.load(fp)
        print("Found pickle file {}.".format(created_arguments["pickle_path"]))
    else:
        print("Pickle file not found, csv will be imported now.")
        tracks = read_track_csv(created_arguments)
        print("Finished importing the pickle file.")

    if created_arguments["save_as_pickle"] and not os.path.exists(created_arguments["pickle_path"]):
        print("Save tracks to pickle file.")
        with open(created_arguments["pickle_path"], "wb") as fp:
            pickle.dump(tracks, fp)

    # Read the static info
    try:
        static_info = read_static_info(created_arguments)
    except:
        print("The static info file is either missing or contains incorrect characters.")
        sys.exit(1)

    # Read the video meta
    try:
        meta_dictionary = read_meta_info(created_arguments)
    except:
        print("The video meta file is either missing or contains incorrect characters.")
        sys.exit(1)

    # if created_arguments["visualize"]:
    #     if tracks is None:
    #         print("Please specify the path to the tracks csv/pickle file.")
    #         sys.exit(1)
    #     if static_info is None:
    #         print("Please specify the path to the static tracks csv file.")
    #         sys.exit(1)
    #     if meta_dictionary is None:
    #         print("Please specify the path to the video meta csv file.")
    #         sys.exit(1)
    #     visualization_plot = VisualizationPlot(created_arguments, tracks, static_info, meta_dictionary)
    #     visualization_plot.show()

# print(len(tracks))
################################################################# pick out vehicles that have a following pattern ################################################################
vf_tracks = filter_vf_tracks(tracks)
# print(tracks[0][BBOX][:,0])
# print('number',len(vf_tracks))
############################################## combine these vehicles with their preceding vehicle and compute everything we need ##############################################
V,A,Nu,D,Pr_x_a,pairs = combine_and_compute(vf_tracks,tracks)
# V,A,Nu,D,Pr_x_a,pairs = combine_and_compute_art(vf_tracks,tracks)
# print(len(V))
# print(len(pairs))
################################################################## dynamic least square solver #################################################################
def OV(x, v_max, h_go, h_st): # nonlinear sigmoidal function 'optimal velocity'
    return 0.5 * v_max * (1 + math.erf(10*(x - (h_go + h_st)/2) / (math.pi * (h_go - h_st + 0.001))))

# main function that calculates the cost to be minimized
def optimization(x): # the OV-FTL model
  P_V = "p_v"
  P_X = "p_x"
  P_L = 'p_l'
  X = "x"
  dt = 0.04

  cost0, e_mean, e_var = 0, 0, 0

  # ind = np.random.randint(len(pairs), size=(50))
  num_ind = len(pairs)#len(pairs)

  # for every vehicle pair in the dataset
  for m in range(num_ind):
    pair = pairs[m]
    x_reg = np.copy(pair[X]) # x_ego to be regenerated
    v_reg = np.copy(pair[X_VELOCITY]) # v_ego to be regenerated
    u_reg = np.copy(pair[X_ACCELERATION])
    pr_x = pair[P_X] # x_pr
    pr_v = pair[P_V] # v_pr
    pr_l = pair[P_L] # length of preceding car
    e = np.zeros(x_reg.shape)

    # this loop calculates cost for every frame 
    for i in range(len(x_reg)-1):
      nu = pr_v[i] - v_reg[i] # nu
      s = pr_x[i] - x_reg[i] - pr_l # s(t)
      u_reg[i] = x[0] * nu / s*s + x[1] * (OV(s,x[2],x[3],x[4]) - v_reg[i])
      v_reg[i+1] = v_reg[i] + dt * u_reg[i] # v(t+1) = v(t) + dt * v'(t), where v'(t) is shown in eq. 4.
      x_reg[i+1] = x_reg[i] + dt * v_reg[i+1] # x(t+1) = x(t) + dt * x'(t) ##v_reg[i]

      cost0 += (x_reg[i+1] - pair[X][i+1])**2

    # this loop calculates error for every frame 
    for i in range(1,len(x_reg)-1):
      p_v = (pair[P_X][i] - pair[P_X][i-1]) / dt
      v = (pair[X][i] - pair[X][i-1]) / dt
      nu = p_v - v
      s = pair[P_X][i] - pair[X][i] - pr_l
      u = x[0] * nu / s*s + x[1] * (OV(s,x[2],x[3],x[4]) - v)
      e[i] = (pair[X][i+1] - 2*pair[X][i] + pair[X][i-1]) / dt**2 - u

    e_mean += np.sum(e)**2
    e_var += np.sum(e*e)
  return cost0 + e_mean + e_var

# the optimization starts here 
x_initial = (1,1,40,50,5) # initial guess 
cons = ({"type": "ineq", "fun": lambda x: x[3] - x[4]}) # constraint 'h_go - h_st > 0'
bnds = ((0, 100), (0, 10), (30, 60), (1, 100), (1, 20))
res_x = minimize(optimization, x_initial, bounds=bnds, constraints=cons, method='trust-constr', 
                 tol=1e-10, options={'xtol': 1e-12})#'trust-constr'

print(res_x)
alpha,beta,v_max,h_go,h_st = res_x.x
print("alpha=",alpha,"beta=",beta,"v_max=",v_max,"h_go=",h_go,"h_st=",h_st)

# here I run the function again to get the cost and error from the results
def optimization_2(x):
  P_V = "p_v"
  P_X = "p_x"
  P_L = 'p_l'
  X = "x"
  dt = 0.04

  cost0, e_mean, e_var = 0, 0, 0
  num_samples = 0

  # ind = np.random.randint(len(pairs), size=(50))
  num_ind = len(pairs)#len(pairs)
  for m in range(num_ind):
    pair = pairs[m]
    x_reg = np.copy(pair[X]) # x_ego to be regenerated
    v_reg = np.copy(pair[X_VELOCITY]) # v_ego to be regenerated
    u_reg = np.copy(pair[X_ACCELERATION])
    pr_x = pair[P_X] # x_pr
    pr_v = pair[P_V] # v_pr
    pr_l = pair[P_L] # length of preceding car
    e = np.zeros(x_reg.shape)
    num_samples += len(x_reg)
    for i in range(len(x_reg)-1):
      nu = pr_v[i] - v_reg[i] # nu
      s = pr_x[i] - x_reg[i] - pr_l # s(t)
      u_reg[i] = x[0] * nu / s*s + x[1] * (OV(s,x[2],x[3],x[4]) - v_reg[i])
      v_reg[i+1] = v_reg[i] + dt * u_reg[i] # v(t+1) = v(t) + dt * v'(t), where v'(t) is shown in eq. 4.
      x_reg[i+1] = x_reg[i] + dt * v_reg[i+1] # x(t+1) = x(t) + dt * x'(t) ##v_reg[i]

      cost0 += (x_reg[i+1] - pair[X][i+1])**2

    for i in range(1,len(x_reg)-1):
      p_v = (pair[P_X][i] - pair[P_X][i-1]) / dt
      v = (pair[X][i] - pair[X][i-1]) / dt
      nu = p_v - v
      s = pair[P_X][i] - pair[X][i] - pr_l
      u = x[0] * nu / s*s + x[1] * (OV(s,x[2],x[3],x[4]) - v)
      e[i] = (pair[X][i+1] - 2*pair[X][i] + pair[X][i-1]) / dt**2 - u

    e_mean += np.sum(e)**2
    e_var += np.sum(e*e)
  sum_error = cost0 + e_mean + e_var
  return sum_error, cost0 / num_samples, e_mean / num_samples, e_var / num_samples

sum_error, cost0, e_mean, e_var = optimization_2(res_x.x)
print("cost_0=",cost0,"e_mean=",e_mean,"e_var=",e_var)
#####################################################################DEAP

################################################################## PyGAD

# # 定义目标函数
# def fitness_func(ga_instance, solution, solution_idx):
#     fitness = 1.0 / float(optimization(solution))
#     return fitness

# # 定义约束条件函数
# def feasible(solution):
#     x0, x1, x2, x3, x4, = solution
#     return 0 <= x0 <= 100 and 0 <= x1 <= 10 and 0 <= x2 <= 45 and 0 <= x3 <= 100 and 0 <= x4 <= 100 and x4 <= x3

# # 适应度函数中加入约束条件检查和惩罚
# def eval_func(ga_instance, solution, solution_idx):
#     if feasible(solution):
#         return 1.0 / float(optimization(solution))
#     else:
#         return -1e6  # 惩罚值

# # 定义问题和优化器
# fitness_function = eval_func

# num_generations = 500
# num_parents_mating = 4

# sol_per_pop = 8
# num_genes = 5

# init_range_low = 0
# init_range_high = 50

# parent_selection_type = "sss"
# keep_parents = 1

# crossover_type = "single_point"

# mutation_type = "random"
# mutation_percent_genes = 10


# ga_instance = pygad.GA(num_generations=num_generations,
#                        num_parents_mating=num_parents_mating,
#                        fitness_func=fitness_function,
#                        sol_per_pop=sol_per_pop,
#                        num_genes=num_genes,
#                        init_range_low=init_range_low,
#                        init_range_high=init_range_high,
#                        parent_selection_type=parent_selection_type,
#                        keep_parents=keep_parents,
#                        crossover_type=crossover_type,
#                        mutation_type=None, 
#                        # mutation_type,
#                        mutation_percent_genes=mutation_percent_genes)

# # 运行优化器
# ga_instance.run()

# # 打印最优解
# solution, solution_fitness, solution_idx = ga_instance.best_solution()
# print("Parameters of the best solution : {solution}".format(solution=solution))
# print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

# # prediction = numpy.sum(numpy.array(function_inputs)*solution)
# # print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))
















# # S = D - h * V#- r
# dt = 0.04
# Ax = np.array([[1,dt],[0,1]])
# Bx = np.array([-dt*(h+dt),-dt])
# Dx = np.array([dt,1])
# ################################################################# calculate and visualize error #################################################################
# # Gamma, d_mu, d_sigma = error_calculate_and_vis(pairs,kp,kd,h,Ax,Bx,Dx)
# # print(Gamma, d_mu, d_sigma)
# # with D
# Gamma, d_mu, d_sigma = 2175.3248780081926, -0.007724966249552918, 0.6250671167191522
# d_mu = 0
# # without D
# # Gamma, d_mu, d_sigma = 2175.3248780081926,-0.04344901981169538,0.4648932840201069
# ############################################################### parameters ####################################################
# N_data = 100
# beta = 0.2
# lambda_v = 0.01*N_data/Gamma#0.0006
# lambda_c = 1
# lambda_b = 0.0005
# random_size = 10

# threshold_1 = 40
# threshold_2 = 12
# ############################################################### construct arrays for qp ####################################################
# # P,q,G = construct_qp_matrices(N_data,S,Nu,Pr_x_a,Ax,Bx,Dx,beta,random_size,lambda_v,d_mu,d_sigma)
# x_t, x_t_plus_1, x_t_minus_1, pr_a = [],[],[],[]

# x_index = np.random.randint(1,len(S),N_data*2)

# x_t.append(np.array([S[x_index[0]],Nu[x_index[0]]]))
# x_t_plus_1.append(np.array([S[x_index[0]+1],Nu[x_index[0]]+1]))
# x_t_minus_1.append(np.array([S[x_index[0]-1],Nu[x_index[0]]-1]))
# pr_a.append(np.array([Pr_x_a[x_index[0]]]))

# for i in range(N_data*2):
#   sign = 0
#   for x in x_t:
#     if S[x_index[i]]<x[0]+threshold_1/N_data**0.5/5 and S[x_index[i]]>x[0]-threshold_1/5/N_data**0.5 and Nu[x_index[i]]<x[1]+threshold_2/5/N_data**0.5 and Nu[x_index[i]]>x[1]-threshold_2/5/N_data**0.5:
#       sign = 1
#       break
#   if sign==0:
#     x_t.append(np.array([S[x_index[i]],Nu[x_index[i]]]))
#     x_t_plus_1.append(np.array([S[x_index[i]+1],Nu[x_index[i]]+1]))
#     x_t_minus_1.append(np.array([S[x_index[i]-1],Nu[x_index[i]]-1]))
#     pr_a.append(np.array([Pr_x_a[x_index[i]]]))

#   if len(x_t)==N_data:
#      break

# def QP(x_t,x_t_plus_1,x_t_minus_1,pr_a):
#   N_data = len(x_t)
#   P = np.zeros((N_data,N_data))
#   q = np.zeros((N_data))

#   for i in range(N_data):
#     # a = x_t_plus_1[i] - Ax @ x_t[i] - Dx * pr_a[i]
#     a = x_t[i] - Ax @ x_t_minus_1[i] - Dx * pr_a[i]
#     b_all = []
#     for j in range(N_data):
#       # b = 2 * beta * (1 + x_t[j] @ Ax @ x_t_minus_1[i]) * x_t[j] @ Bx
#       # b = 2 * beta * (1 + x_t[j] @ x_t_plus_1[i]) * x_t[j] @ Bx
#       b = 2 * beta * (1 + x_t[j] @ x_t[i]) * x_t[j] @ Bx
#       b_all.append(b)

#     P_ = np.zeros((N_data,N_data))
#     for m in range(N_data):
#       for n in range(N_data):
#         P_[m][n] = b_all[m] * b_all[n] * Bx.T @ Bx

#     # previous
#     q_ = np.zeros((N_data))
#     for m in range(N_data):
#       q_[m] = 2 * b_all[m] * a @ Bx

#     # current
#     # q_ = np.zeros((N_data))
#     # w = norm.rvs(0, d_sigma, size=random_size)
#     # for j in range(random_size):
#     #   q_1 = np.zeros((N_data))
#     #   for m in range(N_data):
#     #     q_1[m] = 2 * b_all[m] * (a + w[j]*Bx) @ Bx
#     #   q_ = q_ + q_1/random_size

#     P = P + P_
#     q = q + q_

#   V = np.zeros((N_data,N_data))
#   for m in range(N_data):
#     for n in range(N_data):
#       V[m][n] = -kernel(x_t[m],x_t[n])

#   P_old = P
#   P = P - lambda_v * V

#   G = np.zeros((N_data,N_data))
#   for m in range(N_data):
#     for n in range(N_data):
#       sum = 0
#       # w = norm.rvs(d_mu, d_sigma, size=random_size)
#       w = norm.rvs(0, d_sigma, size=random_size)
#       for i in range(random_size):
#         sum += kernel(x_t[n],(x_t_plus_1[m]+w[i]*Bx))/random_size
#         # sum += kernel(x_t[n],(x_t_plus_1[m]))/random_size
#       G[m][n] = sum - kernel(x_t[n],x_t[m])
 
#   #############################################
#   # 2n*2n without W
#   #############################################
#   # P_c = np.zeros((2*N_data,2*N_data))
#   # P_c[:N_data,:N_data] = P
#   # P_c[N_data:,N_data:] = lambda_c*np.eye(N_data)

#   # q_c = np.zeros(2*N_data)
#   # q_c[:N_data] = q

#   # G_c = np.zeros((3*N_data,2*N_data))
#   # G_c[:N_data,:N_data] = G
#   # g = -np.eye(N_data)
#   # G_c[:N_data,N_data:] = g
#   # G_c[N_data:2*N_data,N_data:] = g
#   # G_c[2*N_data:3*N_data,:N_data] = V

#   #############################################
#   # 3n*3n with W
#   #############################################
#   P_c = np.zeros((3*N_data,3*N_data))
#   P_c[:N_data,:N_data] = P
#   P_c[N_data:2*N_data,N_data:2*N_data] = lambda_c*np.eye(N_data)
#   P_c[2*N_data:3*N_data,2*N_data:3*N_data] = -lambda_b*V

#   P_o = np.zeros((3*N_data,3*N_data))
#   P_o[:N_data,:N_data] = P_old
#   P_o[N_data:2*N_data,N_data:2*N_data] = lambda_c*np.eye(N_data)
#   P_o[2*N_data:3*N_data,2*N_data:3*N_data] = -lambda_b*V

#   q_c = np.zeros(3*N_data)
#   q_c[:N_data] = q

#   G_c = np.zeros((4*N_data,3*N_data))
#   G_c[:N_data,:N_data] = G
#   g = -np.eye(N_data)
#   G_c[:N_data,N_data:2*N_data] = g
#   G_c[:N_data,2*N_data:3*N_data] = -V
#   G_c[N_data:2*N_data,N_data:2*N_data] = g
#   G_c[2*N_data:3*N_data,:N_data] = V
#   G_c[3*N_data:4*N_data,2*N_data:3*N_data] = V

#   #############################################
#   path = os.path.abspath(os.path.dirname(__file__)) + "\\"
#   np.savetxt(path + 'P_.csv', P_o, delimiter = ',')
#   np.savetxt(path + 'P.csv', P_c, delimiter = ',')
#   np.savetxt(path + 'q.csv', q_c, delimiter = ',')
#   np.savetxt(path + 'G.csv', G_c, delimiter = ',')
#   np.save(path + 'x_t.npy', x_t)
#   print(N_data)
#   # print(path)

# def QP_new(x_t,x_t_plus_1,pr_a):
#   N_data = len(x_t)
#   P = np.zeros((N_data,N_data))
#   q = np.zeros((N_data))
#   noise = []
#   for i in range(N_data):
#     w = norm.rvs(0, d_sigma, size=random_size)
#     noise.append(w)

#   for i in range(N_data):
#     if random_size == 0:
#        noise[i] = [0]
#     # for w in noise[i]:
#     #   a = x_t_plus_1[i] + w * Bx - Ax @ x_t[i] - Dx * pr_a[i]
#     #   b_all = []
#     #   for j in range(N_data):
#     #     b_aux = 0
#     #     for w2 in noise[i]:
#     #       b_aux += (1 + x_t[j] @ (x_t_plus_1[i] + w2 * Bx)) * x_t[j] / random_size
#     #     b = 2 * beta * b_aux @ Bx
#     #     # b = 2 * beta * (1 + x_t[j] @ (x_t_plus_1[i] + w * Bx)) * x_t[j] @ Bx
#     #     b_all.append(b)

#     #   P_ = np.zeros((N_data,N_data))
#     #   for m in range(N_data):
#     #     for n in range(N_data):
#     #       P_[m][n] = b_all[m] * b_all[n] * Bx.T @ Bx

#     #   q_ = np.zeros((N_data))
#     #   for m in range(N_data):
#     #     q_[m] = 2 * b_all[m] * a @ Bx

#     #   P = P + P_/max(random_size,1)
#     #   q = q + q_/max(random_size,1)

#     a = x_t_plus_1[i] - Ax @ x_t[i] - Dx * pr_a[i]
#     b_all = []
#     for j in range(N_data):
#       b_aux = 0
#       for w2 in noise[i]:
#         b_aux += (1 + x_t[j] @ (x_t_plus_1[i] + w2 * Bx)) * x_t[j] / random_size
#       b = 2 * beta * b_aux @ Bx
#       # b = 2 * beta * (1 + x_t[j] @ (x_t_plus_1[i] + w * Bx)) * x_t[j] @ Bx
#       b_all.append(b)

#     P_ = np.zeros((N_data,N_data))
#     for m in range(N_data):
#       for n in range(N_data):
#         P_[m][n] = b_all[m] * b_all[n] * Bx.T @ Bx

#     q_ = np.zeros((N_data))
#     for m in range(N_data):
#       q_[m] = 2 * b_all[m] * a @ Bx

#     P = P + P_
#     q = q + q_

#   V = np.zeros((N_data,N_data))
#   for m in range(N_data):
#     for n in range(N_data):
#       V[m][n] = -kernel(x_t[m],x_t[n])

#   P_old = P
#   P = P - lambda_v * V

#   G = np.zeros((N_data,N_data))
#   for m in range(N_data):
#     for n in range(N_data):
#       sum = 0
#       # w = norm.rvs(0, d_sigma, size=random_size)
#       for w in noise[m]:
#         sum += kernel(x_t[n],(x_t_plus_1[m]+w*Bx))
#       G[m][n] = sum/max(random_size,1) - kernel(x_t[n],x_t[m])

#   #############################################
#   # 3n*3n with W
#   #############################################
#   P_c = np.zeros((3*N_data,3*N_data))
#   P_c[:N_data,:N_data] = P
#   P_c[N_data:2*N_data,N_data:2*N_data] = lambda_c*np.eye(N_data)
#   P_c[2*N_data:3*N_data,2*N_data:3*N_data] = -lambda_b*V

#   P_o = np.zeros((3*N_data,3*N_data))
#   P_o[:N_data,:N_data] = P_old
#   P_o[N_data:2*N_data,N_data:2*N_data] = lambda_c*np.eye(N_data)
#   P_o[2*N_data:3*N_data,2*N_data:3*N_data] = -lambda_b*V

#   q_c = np.zeros(3*N_data)
#   q_c[:N_data] = q

#   G_c = np.zeros((4*N_data,3*N_data))
#   G_c[:N_data,:N_data] = G
#   g = -np.eye(N_data)
#   G_c[:N_data,N_data:2*N_data] = g
#   G_c[:N_data,2*N_data:3*N_data] = -V
#   G_c[N_data:2*N_data,N_data:2*N_data] = g
#   G_c[2*N_data:3*N_data,:N_data] = V
#   G_c[3*N_data:4*N_data,2*N_data:3*N_data] = V

#   #############################################
#   path = os.path.abspath(os.path.dirname(__file__)) + "\\"
#   np.savetxt(path + 'P_.csv', P_o, delimiter = ',')
#   np.savetxt(path + 'P.csv', P_c, delimiter = ',')
#   np.savetxt(path + 'q.csv', q_c, delimiter = ',')
#   np.savetxt(path + 'G.csv', G_c, delimiter = ',')
#   np.save(path + 'x_t.npy', x_t)
#   print(N_data)


# QP_new(x_t,x_t_plus_1,pr_a)
# # QP(x_t,x_t_plus_1,x_t_minus_1,pr_a)

# ################################################################# scatter #########################################################
# def scatter_plot():
#   fig, axes = plt.subplots()
#   plt.scatter(S, Nu, s = 10)
#   # plt.scatter(D, Nu, s = 10)
#   # for x in x_t:
#   #   plt.scatter(x[0], x[1], s = 30, c = 20)
#   plt.xticks(np.arange(-20, 25, 5),fontproperties = 'Times New Roman', size = 24)
#   plt.yticks(np.arange(-4, 5, 1),fontproperties = 'Times New Roman', size = 24)
#   plt.xlabel('spacing error',fontdict={'family' : 'Times New Roman', 'size': 28})
#   plt.ylabel('relative velocity',fontdict={'family' : 'Times New Roman', 'size': 28})
#   # axes.set_xticks(np.linspace(-35,20,221))
#   # axes.set_yticks(np.linspace(-7,7,201))
#   axes.grid(alpha=0.3)
#   plt.show()

# # scatter_plot()
# ################################################################# quadratic value funtion  #########################################################
# r = 2.5
# c = 10
# d = 1

# X_idare = np.array([[0.0664810659857291,0.338019400571855],[0.338019400571855,10.2089874277763]])
# # X_idare = np.array([[0.0711926870904237,0.372021429146792],[0.372021429146792,10.1880609946552]])

# # vf_plot(x_t,[],N_data,X_idare,function='quad',three_d=1,contour=1)

