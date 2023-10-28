from data_management.functions import *

X = np.linspace(-20,20,10)
Y = np.linspace(-4,4,10)

Grid, syn_x, S, nu = [],[],[],[]
for x in X:
  for y in Y:
    Grid.append(np.array([x,y]))

data_per_tra = 200

d_mu, d_sigma = 0, 0.6250671167191522
kp, kd, h= 0.006354857329713133, 0.16748344692513145, 0.8903920819008573
dt = 0.04
Ax = np.array([[1,dt],[0,1]])
Bx = np.array([-dt*(h+dt),-dt])
Dx = np.array([dt,1])

for s in Grid:
  x_initial = s
  syn_x.append(x_initial)
  S.append(x_initial[0])
  nu.append(x_initial[1])
  x_old = x_initial

  for _ in range(data_per_tra):
    x_new = dynamics(x_old,Ax,Bx,kp,kd,d_sigma)
    syn_x.append(x_new)
    S.append(x_new[0])
    nu.append(x_new[1])
    x_old = x_new

# plt.scatter(S, nu)
# plt.show()

#########################################################################################
N_data = 100
beta = 0.2
lambda_v = 0.006*N_data/2000#0.0006
lambda_c = 10
lambda_b = 0.0005
random_size = 10

threshold_1 = 40
threshold_2 = 12

x_t, x_t_plus_1, x_t_minus_1, pr_a = [],[],[],[]

x_index, x_index_pre = [], np.random.randint(1,len(syn_x)+1,N_data*2)
for index in x_index_pre:
  if index % data_per_tra != 1 and index % data_per_tra != 0:
    x_index.append(index)

x_t.append(syn_x[x_index[0]])
x_t_plus_1.append(syn_x[x_index[0]+1])
x_t_minus_1.append(syn_x[x_index[0]-1])

for i in range(len(x_index)):
  sign = 0
  for x in x_t:
    if syn_x[x_index[i]][0]<x[0]+threshold_1/N_data**0.5/5 and syn_x[x_index[i]][0]>x[0]-threshold_1/5/N_data**0.5 and syn_x[x_index[i]][1]<x[1]+threshold_2/5/N_data**0.5 and syn_x[x_index[i]][1]>x[1]-threshold_2/5/N_data**0.5:
      sign = 1
      break
  if sign==0:
    x_t.append(syn_x[x_index[i]])
    x_t_plus_1.append(syn_x[x_index[i]+1])
    x_t_minus_1.append(syn_x[x_index[i]-1])

  if len(x_t)==N_data:
     break

def QP(x_t,x_t_plus_1,x_t_minus_1):
  N_data = len(x_t)
  P = np.zeros((N_data,N_data))
  q = np.zeros((N_data))

  for i in range(N_data):
    a = x_t_plus_1[i] - Ax @ x_t[i]# - Dx * pr_a[i]
    b_all = []
    for j in range(N_data):
      # b = 2 * beta * (1 + x_t[j] @ Ax @ x_t_minus_1[i]) * x_t[j] @ Bx
      b = 2 * beta * (1 + x_t[j] @ x_t_plus_1[i]) * x_t[j] @ Bx
      b_all.append(b)

    P_ = np.zeros((N_data,N_data))
    for m in range(N_data):
      for n in range(N_data):
        P_[m][n] = b_all[m] * b_all[n] * Bx.T @ Bx

    q_ = np.zeros((N_data))
    for m in range(N_data):
      q_[m] = 2 * b_all[m] * a @ Bx

    P = P + P_
    q = q + q_

  P_V = np.zeros((N_data,N_data))
  for m in range(N_data):
    for n in range(N_data):
      P_V[m][n] = kernel(x_t[m],x_t[n])

  P = P + lambda_v * P_V

  G = np.zeros((N_data,N_data))
  for m in range(N_data):
    for n in range(N_data):
      sum = 0
      # w = norm.rvs(d_mu, d_sigma, size=random_size)
      w = norm.rvs(0, d_sigma, size=random_size)
      for i in range(random_size):
        sum += kernel(x_t[n],(x_t_plus_1[m]+w[i]*Bx))/random_size
        # sum += kernel(x_t[n],(x_t_plus_1[m]))/random_size
      G[m][n] = sum - kernel(x_t[n],x_t[m])

  V = np.zeros((N_data,N_data))
  for m in range(N_data):
    for n in range(N_data):
      V[m][n] = -kernel(x_t[m],x_t[n])

  P_c = np.zeros((2*N_data,2*N_data))
  P_c[:N_data,:N_data] = P
  P_c[N_data:,N_data:] = lambda_c*np.eye(N_data)####################################

  q_c = np.zeros(2*N_data)
  q_c[:N_data] = q

  G_c = np.zeros((3*N_data,2*N_data))
  G_c[:N_data,:N_data] = G
  g = -np.eye(N_data)
  G_c[:N_data,N_data:] = g
  G_c[N_data:2*N_data,N_data:] = g
  G_c[2*N_data:3*N_data,:N_data] = V

  np.savetxt('P.csv', P_c, delimiter = ',')
  np.savetxt('q.csv', q_c, delimiter = ',')
  np.savetxt('G.csv', G_c, delimiter = ',')
  np.save("x_t.npy",x_t)

  print(N_data)

def QP_new(x_t,x_t_plus_1):
  N_data = len(x_t)
  P = np.zeros((N_data,N_data))
  q = np.zeros((N_data))
  noise = []
  random_size = 1
  for i in range(N_data):
    w = norm.rvs(0, d_sigma, size=random_size)
    noise.append(w)

  for i in range(N_data):
    if random_size == 0:
       noise[i] = [0]
    for w in noise[i]:
      a = x_t_plus_1[i] + w * Bx - Ax @ x_t[i]
      b_all = []
      for j in range(N_data):
        b_aux = 0
        for w2 in noise[i]:
          b_aux += (1 + x_t[j] @ (x_t_plus_1[i] + w2 * Bx)) * x_t[j] / random_size
        b = 2 * beta * b_aux @ Bx
        # b = 2 * beta * (1 + x_t[j] @ (x_t_plus_1[i] + w * Bx)) * x_t[j] @ Bx
        b_all.append(b)

      P_ = np.zeros((N_data,N_data))
      for m in range(N_data):
        for n in range(N_data):
          P_[m][n] = b_all[m] * b_all[n] * Bx.T @ Bx

      q_ = np.zeros((N_data))
      for m in range(N_data):
        q_[m] = 2 * b_all[m] * a @ Bx

      P = P + P_/max(random_size,1)
      q = q + q_/max(random_size,1)

  V = np.zeros((N_data,N_data))
  for m in range(N_data):
    for n in range(N_data):
      V[m][n] = -kernel(x_t[m],x_t[n])

  P_old = P
  P = P - lambda_v * V

  random_size = 10
  for i in range(N_data):
    w = norm.rvs(0, d_sigma, size=random_size)
    noise.append(w)

  G = np.zeros((N_data,N_data))
  for m in range(N_data):
    for n in range(N_data):
      sum = 0
      # w = norm.rvs(0, d_sigma, size=random_size)
      for w in noise[m]:
        sum += kernel(x_t[n],(x_t_plus_1[m]+w*Bx))
      G[m][n] = sum/max(random_size,1) - kernel(x_t[n],x_t[m])

  #############################################
  # 3n*3n with W
  #############################################
  P_c = np.zeros((3*N_data,3*N_data))
  P_c[:N_data,:N_data] = P
  P_c[N_data:2*N_data,N_data:2*N_data] = lambda_c*np.eye(N_data)
  P_c[2*N_data:3*N_data,2*N_data:3*N_data] = -lambda_b*V

  P_o = np.zeros((3*N_data,3*N_data))
  P_o[:N_data,:N_data] = P_old
  P_o[N_data:2*N_data,N_data:2*N_data] = lambda_c*np.eye(N_data)
  P_o[2*N_data:3*N_data,2*N_data:3*N_data] = -lambda_b*V

  q_c = np.zeros(3*N_data)
  q_c[:N_data] = q

  G_c = np.zeros((4*N_data,3*N_data))
  G_c[:N_data,:N_data] = G
  g = -np.eye(N_data)
  G_c[:N_data,N_data:2*N_data] = g
  G_c[:N_data,2*N_data:3*N_data] = -V
  G_c[N_data:2*N_data,N_data:2*N_data] = g
  G_c[2*N_data:3*N_data,:N_data] = V
  G_c[3*N_data:4*N_data,2*N_data:3*N_data] = V

  #############################################
  path = os.path.abspath(os.path.dirname(__file__)) + "\\"
  np.savetxt(path + 'P_.csv', P_o, delimiter = ',')
  np.savetxt(path + 'P.csv', P_c, delimiter = ',')
  np.savetxt(path + 'q.csv', q_c, delimiter = ',')
  np.savetxt(path + 'G.csv', G_c, delimiter = ',')
  np.save(path + 'x_t.npy', x_t)
  print(N_data)


QP_new(x_t,x_t_plus_1)
# QP(x_t,x_t_plus_1,x_t_minus_1)

