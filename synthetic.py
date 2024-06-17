from data_management.functions import *

X = np.linspace(-15, 15, 20)
Y = np.linspace(-3, 3, 20)

Grid, syn_x, Dhw, Nu = [], [], [], []
for x in X:
    for y in Y:
        Grid.append(np.array([x, y]))

data_per_tra = 300

# d_mu, d_sigma = 0, 0.6250671167191522
d_mu, d_sigma = 0, 0.621241544469323
# kp, kd, h = 0.006354857329713133, 0.16748344692513145, 0.8903920819008573
kp, kd, h = 0.017258, 0.617991, 0.913008
dt = 0.04
A_mat = np.array([[1, dt], [0, 1]])
B_mat = np.array([-dt * (h + dt), -dt])
D_mat = np.array([dt, 1])

for s in Grid:
    x_initial = s
    syn_x.append(x_initial)
    Dhw.append(x_initial[0])
    Nu.append(x_initial[1])
    x_old = x_initial

    for _ in range(data_per_tra):
        x_new = dynamics(x_old, A_mat, B_mat, kp, kd, d_sigma)
        syn_x.append(x_new)
        Dhw.append(x_new[0])
        Nu.append(x_new[1])
        x_old = x_new

# plt.scatter(Dhw, Nu, s=1)
# plt.show()

#########################################################################################
N_data = 50
beta = 0.2
lambda_v = 0.01 * N_data / 719.34  #2000
lambda_c = 10
lambda_b = 0.0005
random_size = 25
thres_1 = 30  #40
thres_2 = 8  #12
lambda_U_ker = 0.25  # 0.05
lambda_U_quad = 0.1  # 0.05
con_v1_ker = 2e-1
con_v2_ker = 2e-1
con_v1_quad = 1e-1
con_v2_quad = 1e-1

x_cur, x_next = [], []
ind, ind_pr = [], np.random.randint(1, len(syn_x) + 1, N_data * 4)
for index in ind_pr:
    if index % data_per_tra != 1 and index % data_per_tra != 0:
        ind.append(index)

x_cur.append(syn_x[ind[0]])
x_next.append(syn_x[ind[0] + 1])

for i in range(len(ind)):
    sign = 0
    for x in x_cur:
        if syn_x[ind[i]][0] < x[0] + thres_1 / N_data**0.5 / 5 and syn_x[
                ind[i]][0] > x[0] - thres_1 / 5 / N_data**0.5 and syn_x[
                    ind[i]][1] < x[1] + thres_2 / 5 / N_data**0.5 and syn_x[
                        ind[i]][1] > x[1] - thres_2 / 5 / N_data**0.5:
            sign = 1
            break
    if sign == 0:
        x_cur.append(syn_x[ind[i]])
        x_next.append(syn_x[ind[i] + 1])

    if len(x_cur) == N_data:
        break

print(len(x_cur))
acc = np.zeros(N_data)
for i in range(N_data):
    acc[i] = kp * x_cur[i][0] + kd * x_cur[i][1]

QP_new(x_cur, acc, np.zeros(N_data), d_sigma, random_size, A_mat, B_mat, D_mat,
       beta, lambda_v, lambda_c, lambda_b, lambda_U_ker, con_v1_ker,
       con_v2_ker)

QP_quad(x_cur, acc, np.zeros(N_data), d_sigma, random_size, A_mat, B_mat,
        D_mat, beta, lambda_v, lambda_c, lambda_b, lambda_U_quad, con_v1_quad,
        con_v2_quad)
# def QP_new_syn(x_cur, x_next):
#     N_data = len(x_cur)
#     P_mat = np.zeros((N_data, N_data))
#     q = np.zeros((N_data))
#     # random_size = 20
#     noise = norm.rvs(loc=0, scale=d_sigma, size=(N_data, random_size))
#     # for i in range(N_data):
#     #   w = norm.rvs(loc=0, scale=d_sigma, size=random_size)
#     #   noise.append(w)

#     for i in range(N_data):
#         for w in noise[i, :]:
#             a = x_next[i] + w * B_mat - A_mat @ x_cur[i]
#             b_all = []
#             for j in range(N_data):
#                 b_aux = 0
#                 for w2 in noise[i, :]:
#                     b_aux += (1 + x_cur[j] @ (x_next[i] + w2 * B_mat)
#                               ) * x_cur[j] / random_size
#                 b = 2 * beta * b_aux @ B_mat
#                 b_all.append(b)

#             P_ = np.zeros((N_data, N_data))
#             for m in range(N_data):
#                 for n in range(N_data):
#                     P_[m][n] = b_all[m] * b_all[n] * B_mat.T @ B_mat

#             q_ = np.zeros((N_data))
#             for m in range(N_data):
#                 q_[m] = 2 * b_all[m] * a @ B_mat

#             P_mat = P_mat + P_ / max(random_size, 1)
#             q = q + q_ / max(random_size, 1)

#     V_mat = np.zeros((N_data, N_data))
#     for m in range(N_data):
#         for n in range(N_data):
#             V_mat[m][n] = -kernel(x_cur[m], x_cur[n])

#     P_old = P_mat
#     P_mat = P_mat - lambda_v * V_mat

#     G_mat = np.zeros((N_data, N_data))
#     for m in range(N_data):
#         for n in range(N_data):
#             sum = 0
#             # w = norm.rvs(0, d_sigma, size=random_size)
#             for w in noise[m, :]:
#                 sum += kernel(x_cur[n], (x_next[m] + w * B_mat))
#             G_mat[m][n] = sum / max(random_size, 1) - kernel(
#                 x_cur[n], x_cur[m])

#     #############################################
#     # 3n*3n with W
#     #############################################
#     P_c = np.zeros((3 * N_data, 3 * N_data))
#     P_c[:N_data, :N_data] = P_mat
#     P_c[N_data:2 * N_data, N_data:2 * N_data] = lambda_c * np.eye(N_data)
#     P_c[2 * N_data:3 * N_data, 2 * N_data:3 * N_data] = -lambda_b * V_mat

#     P_o = np.zeros((3 * N_data, 3 * N_data))
#     P_o[:N_data, :N_data] = P_old
#     P_o[N_data:2 * N_data, N_data:2 * N_data] = lambda_c * np.eye(N_data)
#     P_o[2 * N_data:3 * N_data, 2 * N_data:3 * N_data] = -lambda_b * V_mat

#     q_c = np.zeros(3 * N_data)
#     q_c[:N_data] = q

#     G_c = np.zeros((4 * N_data, 3 * N_data))
#     G_c[:N_data, :N_data] = G_mat
#     g = -np.eye(N_data)
#     G_c[:N_data, N_data:2 * N_data] = g
#     G_c[:N_data, 2 * N_data:3 * N_data] = -V_mat
#     G_c[N_data:2 * N_data, N_data:2 * N_data] = g
#     G_c[2 * N_data:3 * N_data, :N_data] = V_mat
#     G_c[3 * N_data:4 * N_data, 2 * N_data:3 * N_data] = V_mat

#     #############################################
#     path = os.path.abspath(os.path.dirname(__file__)) + "\\"
#     np.savetxt(path + 'P_.csv', P_o, delimiter=',')
#     np.savetxt(path + 'P.csv', P_c, delimiter=',')
#     np.savetxt(path + 'q.csv', q_c, delimiter=',')
#     np.savetxt(path + 'G.csv', G_c, delimiter=',')
#     np.save(path + 'x_cur.npy', x_cur)
#     print(N_data)
