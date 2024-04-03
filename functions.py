import os
import sys
import pickle
import argparse
import numpy as np

import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
import math
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
import quadprog
import cvxopt
from scipy import linalg as la
from scipy import special
import pygad

from data_management.read_csv import *

def filter_vf_tracks(tracks):
  """
  This method reads the tracks file from highD data.

  :param arguments: the parsed arguments for the program containing the input path for the tracks csv file.
  :return: a list containing all tracks as dictionaries.
  """
  vf_tracks = []

  for track in tracks:
    dhw = track[DHW]
    pr_id = track[PRECEDING_ID][0]-1
    if not(np.all(dhw==0)) and (track[BBOX][0][2]<6) and (track[LANE_ID][-1] == track[LANE_ID][0]) and (tracks[pr_id][LANE_ID][-1] == tracks[pr_id][LANE_ID][0]):
      vf_dhw = dhw[dhw>1]
      vf_dhw = vf_dhw[vf_dhw<50]
      if (np.count_nonzero(vf_dhw) == np.count_nonzero(dhw)) and (tracks[pr_id][BBOX][0][2]<6) and (np.count_nonzero(dhw)>275):
        vf_tracks.append(track)
  return vf_tracks

def OV(x, v_max, h_go, h_st): # nonlinear sigmoidal function 'optimal velocity'
    return 0.5 * v_max * (1 + special.erf(10*(x - (h_go + h_st)/2) / (math.pi * (h_go - h_st))))

def combine_and_compute(vf_tracks,tracks):
  NU = "nu"
  P_X_V = "p_x_v"
  P_X = "p_x"
  P_L = 'p_l'
  X = "x"
  pairs = []#save information of every vehicle pair
  count = 0
  dt = 0.04
  for track in vf_tracks:
    pr_track = tracks[track[PRECEDING_ID][0]-1]
    frame = np.intersect1d(track[FRAME],pr_track[FRAME])
    x = track[BBOX][:,0][:len(frame)]
    x_v = track[X_VELOCITY][:len(frame)]
    x_a = track[X_ACCELERATION][:len(frame)]
    dhw = track[DHW][:len(frame)]
    pr_x = pr_track[BBOX][:,0][-len(frame):]
    pr_x_v = pr_track[X_VELOCITY][-len(frame):]
    # pr_x_a = pr_track[X_VELOCITY][-len(frame):] - pr_track[X_VELOCITY][-len(frame)-1:-1]
    pr_x_a = pr_track[X_ACCELERATION][-len(frame):]

    nu = pr_x_v - x_v

    # x_reg = np.copy(x)
    # x_v_reg = np.copy(x_v)
    # for i in range(len(frame)-1):
    #   x_v_reg[i+1] = x_v_reg[i] + dt * 
    #   x_reg[i+1] = 

    
    if track[LANE_ID][0] < 4:
      x = -x
      x_a = -x_a
      nu = -nu
      x_v = -x_v
      pr_x = -pr_x
      pr_x_a = -pr_x_a
      pr_x_v = -pr_x_v

    pair = {TRACK_ID: pr_track[TRACK_ID]*100+track[TRACK_ID],  
            FRAME: frame,
            X: x,
            X_VELOCITY: x_v,
            X_ACCELERATION: x_a,
            DHW: dhw,
            NU: nu,
            PRECEDING_ID: track[PRECEDING_ID][:len(frame)],
            LANE_ID: track[LANE_ID][:len(frame)],
            P_L: float(pr_track[BBOX][0,2]),
            P_X: pr_x,
            P_X_V: pr_x_v
            }
    pairs.append(pair)

    if count == 0:
      V = x_v
      A = x_a
      Nu = nu
      D = dhw
      Pr_x_a = pr_x_a
      count = 1
    elif np.all(nu>-4) and np.all(nu<4) and np.all(dhw>10) and np.all(dhw<49):
      V = np.concatenate((V,x_v))
      A = np.concatenate((A,x_a))
      Nu = np.concatenate((Nu,nu))
      D = np.concatenate((D,dhw))
      Pr_x_a = np.concatenate((Pr_x_a,pr_x_a))

  # print(pr_x_a)
  return V,A,Nu,D,Pr_x_a,pairs

def combine_and_compute_art(vf_tracks,tracks):
  NU = "nu"
  P_X_V = "p_x_v"
  pairs = []#save information of every vehicle pair
  count = 0
  dt = 0.04

  for track in vf_tracks:
    pr_track = tracks[track[PRECEDING_ID][0]-1]
    frame = np.intersect1d(track[FRAME],pr_track[FRAME])
    x = track[BBOX][:,0][:len(frame)]
    x_v = track[X_VELOCITY][:len(frame)]
    x_a = track[X_ACCELERATION][:len(frame)]
    x_v_art = np.copy(x_v)
    x_a_art = np.copy(x_a)
    for i in range(len(x)-1):
      x_v_art[i+1] = (x[i+1]-x[i]) / dt
      x_a_art[i+1] = (x_v_art[i+1]-x_v_art[i]) / dt

    dhw = track[DHW][:len(frame)]

    pr_x = pr_track[BBOX][:,0][-len(frame):]
    pr_x_v = pr_track[X_VELOCITY][-len(frame):]
    pr_x_a = pr_track[X_ACCELERATION][-len(frame):]
    pr_x_v_art = np.copy(pr_x_v)
    pr_x_a_art = np.copy(pr_x_a)
    for i in range(len(x)-1):
      pr_x_v_art[i+1] = (pr_x[i+1]-pr_x[i]) / dt
      pr_x_a_art[i+1] = (pr_x_v_art[i+1]-pr_x_v_art[i]) / dt

    nu = pr_x_v - x_v
    nu_art = pr_x_v_art - x_v_art
    
    
    if track[LANE_ID][0] < 4:
      x_a = -x_a
      nu = -nu
      x_v = -x_v
      pr_x_a = -pr_x_a
      pr_x_v = -pr_x_v

      x_a_art = -x_a_art
      nu_art = -nu_art
      x_v_art = -x_v_art
      pr_x_a_art = -pr_x_a_art
      pr_x_v_art = -pr_x_v_art

    pair = {TRACK_ID: pr_track[TRACK_ID]*100+track[TRACK_ID],  
            FRAME: frame,
            X_VELOCITY: x_v,
            X_ACCELERATION: x_a,
            DHW: dhw,
            NU: nu,
            PRECEDING_ID: track[PRECEDING_ID][:len(frame)],
            LANE_ID: track[LANE_ID][:len(frame)],
            P_X_V: pr_x_v
            }
    pairs.append(pair)

    # x_v = x_v_art
    # x_a = x_a_art
    # nu = nu_art
    # dhw = dhw
    # pr_x_a = pr_x_a_art

    if count == 0:
      V = x_v
      A = x_a
      Nu = nu
      D = dhw
      Pr_x_a = pr_x_a
      count = 1
    elif np.all(nu>-4) and np.all(nu<4) and np.all(dhw>10) and np.all(dhw<49):
      V = np.concatenate((V,x_v))
      A = np.concatenate((A,x_a))
      Nu = np.concatenate((Nu,nu))
      D = np.concatenate((D,dhw))
      Pr_x_a = np.concatenate((Pr_x_a,pr_x_a))

  # print(frame)
  # print(len(frame))
  # print(x_v_art)
  # print(x_a_art)

  return V,A,Nu,D,Pr_x_a,pairs

def dynamics(x,Ax,Bx,kp,kd,d_sigma):
  w = norm.rvs(0, d_sigma, size=1)
  return Ax @ x + Bx * (kp*x[0]+kd*x[1]+w)

def error_vis_2(data):
  mu, std = stats.norm.fit(data)

  x = np.linspace(data.min(), data.max(), 100)
  pdf = stats.norm.pdf(x, mu, std)

  plt.hist(data, bins=30, density=True, alpha=0.6, color='b')
  plt.plot(x, pdf, 'r-', lw=2)
  plt.xlabel('values')
  plt.ylabel('Probability')
  plt.title('Histogram : $\mu$=' + str(round(mu,4)) + ' $\sigma=$'+str(round(std,4)))
  plt.show()

def error_vis(x):
  n, bins, patches = plt.hist(x, 100, density=1, alpha=0.75)
  y = norm.pdf(bins, np.mean(x), np.std(x))# fit normal distribution  

  plt.grid(True)
  plt.plot(bins, y, 'r--')
  plt.xlim((-2, 2))
  plt.ylim((0, 1))
  plt.xticks(np.arange(-2, 2.01, 1),fontproperties = 'Times New Roman', size = 28)
  plt.yticks(np.arange(0, 1.01, 0.2),fontproperties = 'Times New Roman', size = 28)
  plt.xlabel('values',fontdict={'family' : 'Times New Roman', 'size': 32})
  plt.ylabel('Probability',fontdict={'family' : 'Times New Roman', 'size': 32})
  plt.title('$\sigma=$'+str(round(np.std(x),4)),fontproperties = 'Times New Roman', size = 30)
  plt.show()

def error_calculate_and_vis(pairs,kp,kd,h,Ax,Bx,Dx):
  NOISE = "noise"
  ERROR = "error"
  NU = "nu"
  P_X_V = "p_x_v"   
  E = "relative_pos"
  count_2 = 0
  count_3 = 0
  for pair in pairs:
    duration = len(pair[FRAME])
    pair[E] = pair[DHW] - h * pair[X_VELOCITY]# - r
    count = 0
    
    for j in range(duration-1):
      error = np.array([[pair[E][j+1]],[pair[NU][j+1]]]) - Ax@np.array([[pair[E][j]],[pair[NU][j]]]) - Bx * (kp*pair[E][j]+kd*pair[NU][j]) - Dx*(pair[P_X_V][j+1]-pair[P_X_V][j])
      disturbance = pair[P_X_V][j+1]-pair[P_X_V][j]
      if count_2 == 0:
        error_all = error
        disturbance_all = disturbance
        count_2 = 1
      else:
        error_all = np.hstack((error_all,error))
        disturbance_all = np.hstack((disturbance_all,disturbance))
      
      if count == 0:
        Error = error
        count = 1
      else:
        Error = np.hstack((Error,error))
        
    noise = kp*pair[DHW]-kp*h*pair[X_VELOCITY]+kd*pair[NU]-pair[X_ACCELERATION]
    if count_3 == 0:
      Noise = noise
      count_3 = 1
    else:
      Noise = np.concatenate((Noise,noise))

    pair[ERROR] = Error
    pair[NOISE] = Noise

  # error_mu = [np.mean(error_all[0,:]),np.mean(error_all[1,:])]
  # error_sigma = [np.std(error_all[0,:]),np.std(error_all[1,:])]
  noise_cov = np.cov(Noise)
  n_mu = np.mean(Noise)
  # print(n_mu,noise_cov)

  #print('noise_cov',noise_cov)
  #print('error_mu',error_mu,'error_sigma',error_sigma)

  Gamma = (noise_cov * Bx@Bx)**(-1)
  # print('Gamma',Gamma)

  mu = np.mean(error_all[0,:])
  sigma = np.std(error_all[0,:])

  filter = (error_all[0,:]>mu-3*sigma) & (error_all[0,:]<mu+3*sigma)
  error_all = np.vstack((error_all[0,:][filter],error_all[1,:][filter]))

  mu_dis = np.mean(disturbance_all)
  sigma_dis = np.std(disturbance_all)
  filter_dis = (disturbance_all>mu_dis-3*sigma_dis) & (disturbance_all<mu_dis+3*sigma_dis)
  # disturbance_all = disturbance_all[filter_dis]
  # error_vis_2(disturbance_all)
  # print(disturbance_all.shape)
  # error_vis(error_all[0,:])
  # error_vis(error_all[1,:])
  error_vis(error_all[0,:]/Bx[0])
  error_vis(error_all[1,:]/Bx[1])

  # B1d = error_all[0,:]/2 + error_all[1,:]*Bx[0,0]/(2*Bx[1,0])
  # B2d = error_all[0,:]*Bx[1,0]/(2*Bx[0,0]) + error_all[1,:]/2
  # d = (c*error_all[0,:]/(Bx[0]) + error_all[1,:]/(Bx[1]))*0.7
  d = (error_all[0,:]/(Bx[0]) + error_all[1,:]/(Bx[1]))*0.6
  # d_mu = c*0.5*np.mean(error_all[0,:])+np.mean(error_all[1,:])*0.5
  # d_sigma = np.std(error_all[1,:])
  d_mu = np.mean(d)
  d_sigma = np.std(d)
  # print('d_mu',d_mu,'d_sigma',d_sigma)
  # print('d_mu',np.mean(d),'d_sigma',np.std(d))
  error_vis(d)
  # error_vis(B1d)
  # error_vis(B2d)
  return Gamma, d_mu, d_sigma

def quadprog_solve_qp(P, q, G, h, A=None, b=None):
  qp_G = .5 * (P + P.T)   # make sure P is symmetric
  qp_a = -q
  if A is not None:
    qp_C = -np.vstack([A, G]).T
    qp_b = -np.hstack([b, h])
    meq = A.shape[0]
  else:  # no equality constraint
    qp_C = -G.T
    qp_b = -h
    meq = 0
  return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]

def cvxopt_solve_qp(P, q, G, h, A=None, b=None):
  P = .5 * (P + P.T)  # make sure P is symmetric
  args = [cvxopt.matrix(P), cvxopt.matrix(q)]
  args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
  if A is not None:
      args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
  sol = cvxopt.solvers.qp(*args)
  if 'optimal' not in sol['status']:
      return None
  return np.array(sol['x']).reshape((P.shape[1],))

def kernel(x1,x2):
  return (1+x1@x2)**2###############

def value_func(x,x_t,alphas,N_data):
  sum = 0
  for i in range(N_data):
    sum += alphas[i]*kernel(x,x_t[i])
  return sum

def quad_value_func(x,X):
  return x.T@X@x

# def derivative_v(x,x_t,alphas,N_data):
#   sum = 0
#   for i in range(N_data):
#     sum += alphas[i]*2*(1+x_t[i]@x)*x_t[i].T
#   return sum

# def pi_hat(x,x_all,alphas,N_data,Ax,Bx,beta):
#   de = derivative_v(Ax@x,x_all,alphas,N_data)
#   return -beta*de@Bx

def vf_plot(x_t,alphas,N_data,X_idare=0,function='kernel',three_d=1,contour=1):
  # X = np.linspace(-35,20,200) #0.5 20*20
  # Y = np.linspace(-7,7,200) #0.1 20*20
  n = 100
  X = np.linspace(-20,20,n) #0.5 20*20
  Y = np.linspace(-5,5,n) #0.1 20*20
  X,Y = np.meshgrid(X,Y)
  Z = np.zeros((n,n))

  if function == 'kernel':
    for i in range(n):
      for j in range(n):
        Z[j,i] = value_func(np.array([X[0,i],Y[j,0]]),x_t,alphas,N_data)
  if function == 'quad':
    for i in range(n):
      for j in range(n):
        Z[j,i] = quad_value_func(np.array([X[0,i],Y[j,0]]),X_idare)

  if three_d==1:
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.zaxis.set_major_formatter('{x:.02f}')
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

  if contour==1:
    # plt.contourf(X,Y,Z)

    # for x in x_t:
    #   if x[0]<20:
    #     plt.scatter(x[0], x[1], s = 30, c = 'b')
    C=plt.contour(X,Y,Z,levels=[10**i for i in range(-4,8)])
    plt.clabel(C, inline=True, fontsize=10)
    plt.show()
  
  # print(value_func(np.array([-20,0]),x_t,alphas,N_data))