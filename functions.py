import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
from matplotlib import cm
from scipy.stats import norm

from data_management.read_csv import *


def filter_vf_tracks(tracks):
    """
    This method filters the vehicle-following behavior from input tracks.

    :param arguments: input tracks.
    :return: vf_tracks -  a list containing tracks contaning vehicle-following behavior as dictionaries.
    """
    # Declare and initialize the vf_tracks
    vf_tracks = []
    for track in tracks:
        dhw = track[DHW]
        pr_id = track[PRECEDING_ID][0] - 1
        if not (np.all(dhw == 0)) and (track[BBOX][0][2] < 6) and (
                track[LANE_ID][-1]
                == track[LANE_ID][0]) and (tracks[pr_id][LANE_ID][-1]
                                           == tracks[pr_id][LANE_ID][0]):
            vf_dhw = dhw[dhw > 1]
            vf_dhw = vf_dhw[vf_dhw < 50]
            if (np.count_nonzero(vf_dhw) == np.count_nonzero(dhw)) and (
                    tracks[pr_id][BBOX][0][2] < 6) and (np.count_nonzero(dhw)
                                                        > 275):
                vf_tracks.append(track)
    return vf_tracks


def extract_features(vf_tracks, tracks):
    NU = "nu"
    P_V = "p_v"
    P_X = "p_x"
    P_L = 'p_l'
    X = "x"
    pairs = []  #save information of every vehicle pair
    count = 0
    frame_skip = 10
    for track in vf_tracks:
        pr_track = tracks[track[PRECEDING_ID][0] - 1]
        frame = np.intersect1d(track[FRAME], pr_track[FRAME])  #[frame_skip:]
        x = track[BBOX][:, 0][frame_skip:len(frame)]
        v = track[X_VELOCITY][frame_skip:len(frame)]
        a = track[X_ACCELERATION][frame_skip:len(frame)]
        dhw = track[DHW][frame_skip:len(frame)]
        pr_x = pr_track[BBOX][:, 0][-len(frame) + frame_skip:]
        pr_v = pr_track[X_VELOCITY][-len(frame) + frame_skip:]
        # pr_x_a = pr_track[X_VELOCITY][-len(frame):] - pr_track[X_VELOCITY][-len(frame)-1:-1]
        pr_a = pr_track[X_ACCELERATION][-len(frame) + frame_skip:]

        nu = pr_v - v

        if track[LANE_ID][0] < 4:
            x = -x
            a = -a
            nu = -nu
            v = -v
            pr_x = -pr_x
            pr_a = -pr_a
            pr_v = -pr_v

        pair = {
            TRACK_ID: pr_track[TRACK_ID] * 100 + track[TRACK_ID],
            FRAME: frame,
            X: x,
            X_VELOCITY: v,
            X_ACCELERATION: a,
            DHW: dhw,
            NU: nu,
            PRECEDING_ID: track[PRECEDING_ID][:len(frame)],
            LANE_ID: track[LANE_ID][:len(frame)],
            P_L: float(pr_track[BBOX][0, 2]),
            P_X: pr_x,
            P_V: pr_v
        }
        pairs.append(pair)

        if count == 0:
            Vel = v
            Acc = a
            Nu = nu
            Dhw = dhw
            Pr_x_a = pr_a
            count = 1
        elif np.all(nu > -4) and np.all(nu < 4) and np.all(
                dhw > 10) and np.all(dhw < 49):
            Vel = np.concatenate((Vel, v))
            Acc = np.concatenate((Acc, a))
            Nu = np.concatenate((Nu, nu))
            Dhw = np.concatenate((Dhw, dhw))
            Pr_x_a = np.concatenate((Pr_x_a, pr_a))

    # print(pr_x_a)
    return Vel, Acc, Nu, Dhw, Pr_x_a, pairs


def extract_features_gen(vf_tracks, tracks):
    NU = "nu"
    P_X_V = "p_x_v"
    pairs = []  #save information of every vehicle pair
    count = 0
    dt = 0.04
    frame_skip = 10

    for track in vf_tracks:
        pr_track = tracks[track[PRECEDING_ID][0] - 1]
        frame = np.intersect1d(track[FRAME], pr_track[FRAME])

        x = track[BBOX][:, 0][frame_skip:len(frame)]
        v = track[X_VELOCITY][frame_skip:len(frame)]
        a = track[X_ACCELERATION][frame_skip:len(frame)]
        dhw = track[DHW][frame_skip:len(frame)]
        pr_x = pr_track[BBOX][:, 0][-len(frame) + frame_skip:]
        pr_v = pr_track[X_VELOCITY][-len(frame) + frame_skip:]
        pr_a = pr_track[X_ACCELERATION][-len(frame) + frame_skip:]

        v_art = np.copy(v)
        a_art = np.copy(a)
        for i in range(len(x) - 1):
            v_art[i + 1] = (x[i + 1] - x[i]) / dt
            a_art[i + 1] = (v_art[i + 1] - v_art[i]) / dt

        pr_v_art = np.copy(pr_v)
        pr_a_art = np.copy(pr_a)
        for i in range(len(x) - 1):
            pr_v_art[i + 1] = (pr_x[i + 1] - pr_x[i]) / dt
            pr_a_art[i + 1] = (pr_v_art[i + 1] - pr_v_art[i]) / dt

        nu = pr_v - v
        nu_art = pr_v_art - v_art

        if track[LANE_ID][0] < 4:
            a = -a
            nu = -nu
            v = -v
            pr_a = -pr_a
            pr_v = -pr_v

            a_art = -a_art
            nu_art = -nu_art
            v_art = -v_art
            pr_a_art = -pr_a_art
            pr_v_art = -pr_v_art

        v = v_art
        a = a_art
        nu = nu_art
        dhw = dhw
        pr_a = pr_a_art

        pair = {
            TRACK_ID: pr_track[TRACK_ID] * 100 + track[TRACK_ID],
            FRAME: frame,
            X_VELOCITY: v,
            X_ACCELERATION: a,
            DHW: dhw,
            NU: nu,
            PRECEDING_ID: track[PRECEDING_ID][:len(frame)],
            LANE_ID: track[LANE_ID][:len(frame)],
            P_X_V: pr_v
        }
        pairs.append(pair)

        if count == 0:
            Vel = v
            Acc = a
            Nu = nu
            Dhw = dhw
            Pr_a = pr_a
            count = 1
        elif np.all(nu > -4) and np.all(nu < 4) and np.all(
                dhw > 10) and np.all(dhw < 49):
            Vel = np.concatenate((Vel, v))
            Acc = np.concatenate((Acc, a))
            Nu = np.concatenate((Nu, nu))
            Dhw = np.concatenate((Dhw, dhw))
            Pr_a = np.concatenate((Pr_a, pr_a))

    # print(frame)
    # print(len(frame))
    # print(v_art)
    # print(a_art)

    return Vel, Acc, Nu, Dhw, Pr_a, pairs


def dynamics(x, Ax, Bx, kp, kd, d_sigma):
    w = 0  #norm.rvs(0, d_sigma, size=1)
    return Ax @ x + Bx * (kp * x[0] + kd * x[1] + w)


def error_vis(x):
    n, bins, patches = plt.hist(x, 100, density=1, alpha=0.75)
    y = norm.pdf(bins, np.mean(x), np.std(x))  # fit normal distribution

    plt.grid(True)
    plt.plot(bins, y, 'r--')
    plt.xlim((-2, 2))
    plt.ylim((0, 1))
    plt.xticks(np.arange(-2, 2.01, 1),
               fontproperties='Times New Roman',
               size=28)
    plt.yticks(np.arange(0, 1.01, 0.2),
               fontproperties='Times New Roman',
               size=28)
    plt.xlabel('values', fontdict={'family': 'Times New Roman', 'size': 32})
    plt.ylabel('Probability',
               fontdict={
                   'family': 'Times New Roman',
                   'size': 32
               })
    plt.title('$\sigma=$' + str(round(np.std(x), 4)),
              fontproperties='Times New Roman',
              size=30)
    plt.show()


def error_vis_3(data):
    hist, bins = np.histogram(data, bins=100, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    def normal_distribution(x, mean, std_dev):
        return norm.pdf(x, loc=mean, scale=std_dev)

    params, covariance = curve_fit(normal_distribution,
                                   bin_centers,
                                   hist,
                                   p0=[np.mean(data),
                                       np.std(data)])

    plt.hist(data,
             bins=200,
             density=True,
             alpha=0.6,
             color='royalblue',
             label='Data Histogram')

    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 1000)
    fit_mean, fit_std_dev = params
    pdf = normal_distribution(x, fit_mean, fit_std_dev)
    plt.plot(x, pdf, 'r', linewidth=2, label='Fitted Normal Distribution')

    plt.grid(True)
    plt.xlabel('Value', fontdict={'family': 'Times New Roman', 'size': 32})
    plt.ylabel('Probability Density',
               fontdict={
                   'family': 'Times New Roman',
                   'size': 32
               })
    plt.xlim((-1.5, 1.5))
    plt.ylim((0, 1.4))
    plt.xticks(np.arange(-1.5, 1.51, 0.75),
               fontproperties='Times New Roman',
               size=28)
    plt.yticks(np.arange(0, 1.41, 0.2),
               fontproperties='Times New Roman',
               size=28)
    # plt.title(r'$\xi_{i,1},\sigma=$'+str(round(np.std(data),4)),size = 15)#,fontproperties = 'Times New Roman', size = 30)
    plt.title(r'$\xi_{i}$', size=32)
    plt.rc('legend', fontsize=28)
    # plt.legend()
    plt.show()


def error_cal(pairs, kp, kd, h, Ax, Bx, Dx):
    NOISE = "noise"
    ERROR = "error"
    NU = "nu"
    P_V = "p_v"
    SE = "spacing_error"
    count_2 = 0
    count_3 = 0
    for pair in pairs:
        duration = len(pair[X])
        pair[SE] = pair[DHW] - h * pair[X_VELOCITY]  # - r
        count = 0

        for j in range(duration - 1):
            State_cur = np.array([pair[SE][j + 1], pair[NU][j + 1]])
            State_pre = np.array([pair[SE][j], pair[NU][j]])
            error = State_cur - Ax @ State_pre - Bx * (
                kp * pair[SE][j] + kd * pair[NU][j]) - Dx * (pair[P_V][j + 1] -
                                                             pair[P_V][j])
            # disturbance = pair[P_V][j+1]-pair[P_V][j]
            if count_2 == 0:
                error_all = np.reshape(error, [2, 1])
                count_2 = 1
            else:
                error_all = np.hstack((error_all, np.reshape(error, [2, 1])))

            if count == 0:
                Error = np.reshape(error, [2, 1])
                count = 1
            else:
                Error = np.hstack((Error, np.reshape(error, [2, 1])))

        noise = kp * pair[DHW] - kp * h * pair[X_VELOCITY] + kd * pair[
            NU] - pair[X_ACCELERATION]
        if count_3 == 0:
            Noise = noise
            count_3 = 1
        else:
            Noise = np.concatenate((Noise, noise))

        pair[ERROR] = Error
        pair[NOISE] = Noise

    # error_mu = [np.mean(error_all[0,:]),np.mean(error_all[1,:])]
    # error_sigma = [np.std(error_all[0,:]),np.std(error_all[1,:])]
    noise_cov = np.cov(Noise)
    n_mu = np.mean(Noise)
    # print(n_mu,noise_cov)

    Gamma = (noise_cov * Bx @ Bx)**(-1)
    # print('Gamma',Gamma)

    # print(error_all.shape)
    mu = np.mean(error_all[0, :])
    sigma = np.std(error_all[0, :])
    filter = (error_all[0, :] > mu - 3 * sigma) & (error_all[0, :]
                                                   < mu + 3 * sigma)
    error_all = np.vstack((error_all[0, :][filter], error_all[1, :][filter]))
    # print(error_all[1,:100])

    # error_vis(error_all[0,:])
    # error_vis(error_all[1,:])
    # error_vis(error_all[0,:]/Bx[0])
    # error_vis(error_all[1,:]/Bx[1])
    # error_vis_3(error_all[0,:]/Bx[0])
    # error_vis_3(error_all[1,:]/Bx[1])

    # B1d = error_all[0,:]/2 + error_all[1,:]*Bx[0,0]/(2*Bx[1,0])
    # B2d = error_all[0,:]*Bx[1,0]/(2*Bx[0,0]) + error_all[1,:]/2
    # d = (c*error_all[0,:]/(Bx[0]) + error_all[1,:]/(Bx[1]))*0.7
    d = (error_all[0, :] / (Bx[0]) + error_all[1, :] / (Bx[1])) * 0.6
    # d_mu = c*0.5*np.mean(error_all[0,:])+np.mean(error_all[1,:])*0.5
    # d_sigma = np.std(error_all[1,:])
    d_mu = np.mean(d)
    d_sigma = np.std(d)
    # print('d_mu',d_mu,'d_sigma',d_sigma)
    # print('d_mu',np.mean(d),'d_sigma',np.std(d))
    error_vis_3(d)
    # error_vis(B1d)
    # error_vis(B2d)
    return Gamma, d_mu, d_sigma


def kernel(x1, x2):
    return (1 + x1 @ x2)**2


def value_func(x, x_t, alphas, N_data):
    sum_ = 0
    for i in range(N_data):
        sum_ += alphas[i] * kernel(x, x_t[i])
    return sum_


def quad_value_func(x1, x2):
    return x1.T @ x2 @ x1


def OV(x, v_max, h_go, h_st):
    """
    This function is the nonlinear sigmoidal function 'optimal velocity'
    """
    return 0.5 * v_max * (1 + math.erf(10 * (x - (h_go + h_st) / 2) /
                                       (math.pi * (h_go - h_st + 0.001))))


def vf_plot(x_t,
            alphas,
            alphas_quad,
            N_data,
            X_idare=0,
            function='kernel',
            three_d=1,
            contour=1):
    # x = np.linspace(-35,20,200) #0.5 20*20
    # y = np.linspace(-7,7,200) #0.1 20*20
    n = 100
    x = np.linspace(-20, 20, n)  #0.5 20*20
    y = np.linspace(-5, 5, n)  #0.1 20*20
    x, y = np.meshgrid(x, y)
    z = np.zeros((n, n))
    z_quad = np.zeros((n, n))

    if function == 'kernel':
        for i in range(n):
            for j in range(n):
                z[j, i] = value_func(np.array([x[0, i], y[j, 0]]), x_t, alphas,
                                     N_data)
    if function == 'quad':
        for i in range(n):
            for j in range(n):
                z[j, i] = quad_value_func(np.array([x[0, i], y[j, 0]]),
                                          X_idare)
    for i in range(n):
        for j in range(n):
            z_quad[j, i] = np.array(
                [x[0, i]**2, 2 * x[0, i] * y[j, 0], y[j, 0]**2]) @ alphas_quad

    if three_d == 1:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(x,
                               y,
                               z,
                               cmap=cm.coolwarm,
                               linewidth=0,
                               antialiased=False)
        ax.zaxis.set_major_formatter('{x:.02f}')
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

    if contour == 1:
        for xt in x_t:
            if xt[0] < 20:
                plt.scatter(xt[0], xt[1], s=30, c='b')
        C = plt.contour(x, y, z, levels=[10**i for i in range(-1, 5)])
        plt.clabel(C, inline=True, fontsize=10)

        C_quad = plt.contour(x,
                             y,
                             z_quad,
                             levels=[10**i for i in range(-1, 5)],
                             linestyles='dashdot')
        plt.clabel(C_quad, inline=True, fontsize=10)

        # plt.legend(handles=[l1, l2], labels=['kernel', 'quad'], loc='best')
        plt.grid()
        plt.show()

    # print(value_func(np.array([-20,0]),x_t,alphas,N_data))


def scatter_plot(Se, Nu):
    fig, axes = plt.subplots()
    plt.scatter(Se, Nu, s=10)
    # for x in x_t:
    #   plt.scatter(x[0], x[1], s = 30, c = 20)
    plt.xticks(np.arange(-20, 25, 5),
               fontproperties='Times New Roman',
               size=24)
    plt.yticks(np.arange(-4, 5, 1), fontproperties='Times New Roman', size=24)
    plt.xlabel('spacing error e$_i$ [m]',
               fontdict={
                   'family': 'Times New Roman',
                   'size': 40
               })
    plt.ylabel('relative velocity '
               r'$\nu_i$ [m/s]',
               fontdict={
                   'family': 'Times New Roman',
                   'size': 40
               })
    # axes.set_xticks(np.linspace(-35,20,221))
    # axes.set_yticks(np.linspace(-7,7,201))
    axes.grid(alpha=0.3)
    plt.show()


def QP_new(x_cur, x_next, acc, pr_a, d_sigma, random_size, A_mat, B_mat, D_mat,
           beta, lambda_v, lambda_c, lambda_w, gamma_regulU):
    R = 2.5
    N_data = len(x_cur)
    par_dim = N_data
    var_dim = N_data * 2 + 1
    constr_dim = N_data + N_data * (1 + random_size) + 1 + N_data * (
        1 + random_size)
    P_mat = np.zeros((var_dim, var_dim))
    q = np.zeros(var_dim)
    A_cost = np.zeros((constr_dim, var_dim))
    b_cost = np.zeros((constr_dim, 1))

    noise = norm.rvs(loc=0, scale=d_sigma, size=(N_data, random_size))
    x_next_realiz = np.zeros((N_data, random_size, 2))
    for i in range(N_data):
        for j in range(random_size):
            x_next_realiz[
                i, j, :] = A_mat @ x_cur[i] + B_mat * (acc[i] + noise[i, j])

    V_mat = np.zeros((N_data, N_data))
    for m in range(N_data):
        for n in range(N_data):
            V_mat[m][n] = -kernel(x_cur[m], x_cur[n])

    for i in range(N_data):
        # b for each data
        b = np.zeros(N_data)
        for ii in range(random_size):
            x_next_sample = x_next_realiz[i, ii, :]
            for j in range(N_data):
                b_aux = (1 + x_cur[j] @ x_next_sample) * x_cur[j]
                b[j] += 2 * beta * b_aux @ B_mat / random_size

        for ii in range(random_size):
            x_next_sample = x_next_realiz[i, ii, :]
            a = x_next_sample - A_mat @ x_cur[i] - D_mat * pr_a[i]

            P_mat[:par_dim, :par_dim] += b.reshape((N_data, 1)) @ b.reshape(
                (N_data,
                 1)).T * (B_mat.T @ B_mat + 1 * gamma_regulU) / random_size
            q[:par_dim] += 2 * (a @ B_mat +
                                acc[i] * gamma_regulU) * b / random_size

    # term for c^2
    P_mat[par_dim, par_dim] += lambda_c
    # another regularization
    P_mat[:par_dim, :par_dim] += -lambda_v * V_mat
    P_mat[-par_dim:, -par_dim:] += -lambda_w * V_mat

    for i in range(N_data):
        xx = np.array(
            [x_cur[i][0]**2, 2 * x_cur[i][0] * x_cur[i][1], x_cur[i][1]**2])
        for j in range(random_size):
            x_next_sample = x_next_realiz[i, j, :]
            xx_next = np.array([
                x_next_sample[0]**2, 2 * x_next_sample[0] * x_next_sample[1],
                x_next_sample[1]**2
            ]).reshape((1, 3))

            V_realiz = np.zeros(par_dim)
            for ii in range(N_data):
                V_realiz[ii] = kernel(x_next_sample, x_cur[ii])

            # term for -x'*Pv*x<0
            A_cost[N_data + N_data + i * random_size + j, :par_dim] = -V_realiz
            b_cost[N_data + N_data + i * random_size + j,
                   0] = -0.8e-1 * xx_next @ np.array([1, 0, 1])

            # term for -x'*Pw*x<0
            A_cost[N_data + N_data * (1 + random_size) + 1 + N_data +
                   i * random_size + j,
                   par_dim + 1:par_dim + 1 + par_dim] = -V_realiz
            b_cost[N_data + N_data * (1 + random_size) + 1 + N_data +
                   i * random_size + j,
                   0] = -1e-4 * xx_next @ np.array([1, 0, 1])

        G_ = np.zeros(par_dim)
        for j in range(N_data):
            sum_ = 0
            for ii in range(random_size):
                sum_ += kernel(x_cur[j], x_next_realiz[i, ii, :])
            G_[j] = sum_ / random_size - kernel(x_cur[i], x_cur[i])

        V_ = np.zeros(par_dim)
        for j in range(N_data):
            V_[j] = kernel(x_cur[i], x_cur[j])

        # term for E[Delta V]= V_next - V - c + W <0
        A_cost[i, :par_dim] = G_
        A_cost[i, par_dim] = -1
        A_cost[i, par_dim + 1:par_dim + 1 + par_dim] = V_
        b_cost[i, 0] = -acc[i] * R * acc[i]
        # term for -x'*Pv*x<0
        A_cost[N_data + i, :par_dim] = -V_
        b_cost[N_data + i, 0] = -5e-1 * xx @ np.array([1, 0, 1])
        # term for -c<0
        A_cost[N_data + N_data * (1 + random_size) + 1, par_dim + 1] = -1
        # term for -x'*Pw*x<0
        A_cost[N_data + N_data * (1 + random_size) + 1 + i,
               par_dim + 1:par_dim + 1 + par_dim] = -V_
        b_cost[N_data + N_data * (1 + random_size) + 1 + i,
               0] = -1e-4 * xx @ np.array([1, 0, 1])

    #############################################
    path = os.path.abspath('.') + "\\data_inter\\"
    np.savetxt(path + 'P.csv', P_mat, delimiter=',')
    np.savetxt(path + 'q.csv', q, delimiter=',')
    np.savetxt(path + 'G.csv', A_cost, delimiter=',')
    np.savetxt(path + 'h.csv', b_cost, delimiter=',')
    np.save(path + 'x_cur.npy', x_cur)
    print(N_data)


def QP_quad(x_cur, x_next, acc, pr_a, d_sigma, random_size, A_mat, B_mat,
            D_mat, beta, lambda_v, lambda_c, lambda_w, gamma_regulU):
    R = 2.5
    lambda_c = 0.05
    N_data = len(x_cur)
    par_dim = 3
    var_dim = 7
    constr_dim = N_data + N_data * (1 + random_size) + 1 + N_data * (
        1 + random_size)
    P_mat = np.zeros((var_dim, var_dim))
    q = np.zeros((var_dim, 1))
    A_cost = np.zeros((constr_dim, var_dim))
    b_cost = np.zeros((constr_dim, 1))

    noise = norm.rvs(loc=0, scale=d_sigma, size=(N_data, random_size))
    x_next_realiz = np.zeros((N_data, random_size, 2))
    for i in range(N_data):
        for j in range(random_size):
            x_next_realiz[
                i, j, :] = A_mat @ x_cur[i] + B_mat * (acc[i] + noise[i, j])

    for i in range(N_data):
        xx = np.array(
            [x_cur[i][0]**2, 2 * x_cur[i][0] * x_cur[i][1],
             x_cur[i][1]**2]).reshape((3, 1))

        dV_aux = np.zeros((2, 3))
        for j in range(random_size):
            x_next_sample = x_next_realiz[i, j, :]
            dV_aux += 2 * np.array([[x_next_sample[0], x_next_sample[1], 0],
                                    [0, x_next_sample[0], x_next_sample[1]]
                                    ]) / random_size
        B_mat = B_mat.reshape((2, 1))
        b = -beta * B_mat @ B_mat.T @ dV_aux
        aa = acc[i]
        bb = -beta * B_mat.T @ dV_aux

        for j in range(random_size):
            x_next_sample = x_next_realiz[i, j, :]
            xx_next = np.array([
                x_next_sample[0]**2, 2 * x_next_sample[0] * x_next_sample[1],
                x_next_sample[1]**2
            ]).reshape((3, 1))
            a = x_next_sample - A_mat @ x_cur[i] - D_mat * pr_a[i]

            # double loop
            # dV_aux = 2 * np.array([[x_next_sample[0], x_next_sample[1], 0],
            #                        [0, x_next_sample[0], x_next_sample[1]]])
            # B_mat = B_mat.reshape((2, 1))
            # b = -beta * B_mat @ B_mat.T @ dV_aux

            P_mat[:par_dim, :par_dim] += (
                b.T @ b + bb.T @ bb * gamma_regulU) / random_size
            q[:par_dim] += 2 * (b.T @ a.reshape(
                (2, 1)) + bb.T * aa * gamma_regulU) / random_size
            # term for c^2
            P_mat[par_dim, par_dim] += lambda_c

            P_mat[:par_dim, :par_dim] += lambda_v * (xx @ xx.T +
                                                     xx_next @ xx_next.T) / 2
            P_mat[-par_dim:, -par_dim:] += lambda_w * (xx @ xx.T +
                                                       xx_next @ xx_next.T) / 2
            q[:par_dim] += lambda_v * (xx + xx_next) / 2
            q[-par_dim:] += lambda_w * (xx + xx_next) / 2

    for i in range(N_data):
        xx = np.array(
            [x_cur[i][0]**2, 2 * x_cur[i][0] * x_cur[i][1], x_cur[i][1]**2])
        xx_aux = np.zeros(par_dim).reshape((1, 3))
        for j in range(random_size):
            x_next_sample = x_next_realiz[i, j, :]
            xx_next = np.array([
                x_next_sample[0]**2, 2 * x_next_sample[0] * x_next_sample[1],
                x_next_sample[1]**2
            ]).reshape((1, 3))
            xx_aux += xx_next / random_size

            # term for -x'*Pv*x<0
            A_cost[N_data + N_data + i * random_size + j, :par_dim] = -xx_next
            b_cost[N_data + N_data + i * random_size + j,
                   0] = -0.8e-1 * xx_next @ np.array([1, 0, 1])

            # term for -x'*Pw*x<0
            A_cost[N_data + N_data * (1 + random_size) + 1 + N_data +
                   i * random_size + j,
                   par_dim + 1:par_dim + 1 + par_dim] = -xx_next
            b_cost[N_data + N_data * (1 + random_size) + 1 + N_data +
                   i * random_size + j,
                   0] = -1e-4 * xx_next @ np.array([1, 0, 1])

        # term for E[Delta V]= V_next - V - c + W <0
        A_cost[i, :par_dim] = xx_aux - xx
        A_cost[i, par_dim] = -1
        A_cost[i, par_dim + 1:par_dim + 1 + par_dim] = xx
        b_cost[i, 0] = -acc[i] * R * acc[i]
        # term for -x'*Pv*x<0
        A_cost[N_data + i, :par_dim] = -xx
        b_cost[N_data + i, 0] = -5e-1 * xx @ np.array([1, 0, 1])
        # term for -c<0
        A_cost[N_data + N_data * (1 + random_size) + 1, par_dim + 1] = -1
        # term for -x'*Pw*x<0
        A_cost[N_data + N_data * (1 + random_size) + 1 + i,
               par_dim + 1:par_dim + 1 + par_dim] = -xx
        b_cost[N_data + N_data * (1 + random_size) + 1 + i,
               0] = -1e-4 * xx @ np.array([1, 0, 1])

    #############################################
    path = os.path.abspath('.') + "\\data_inter\\"
    np.savetxt(path + 'P_quad.csv', P_mat, delimiter=',')
    np.savetxt(path + 'q_quad.csv', q, delimiter=',')
    np.savetxt(path + 'A_quad.csv', A_cost, delimiter=',')
    np.savetxt(path + 'B_quad.csv', b_cost, delimiter=',')
    # np.save(path + 'x_cur.npy', x_cur)
    print(N_data)
