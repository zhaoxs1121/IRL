from data_management.functions import *
import scipy.io

path = os.path.abspath(os.path.dirname(__file__)) + "\\"
x_t = np.load(path + 'x_t.npy')
matlab_data = scipy.io.loadmat(path + 'solution.mat')
# x_t = np.load("D:/learn/23SS/guided research/latex/results/4/x_t.npy")
# matlab_data = scipy.io.loadmat('D:/learn/23SS/guided research/latex/results/4/solution.mat')

x = matlab_data['x']
N_data = int(len(x)/3)
alphas = x[:N_data]
vf_plot(x_t,alphas,N_data,[],function='kernel',three_d=1,contour=1)

# X_idare_1 = np.array([[0.0664810659857291,0.338019400571855],[0.338019400571855,10.2089874277763]])
# X_idare = np.array([[0.0711926870904237,0.372021429146792],[0.372021429146792,10.1880609946552]])
# X_idare_2 = np.array([[0.0430753654485005,0.369868386733732],[0.369868386733732,10.1552921819148]])
# X_idare_2 = np.array([[0.158582088201046,0.264327754892437],[0.00693605081833310,24.1840146927528]])
# # vf_plot(x_t,[],N_data,X_idare,function='quad',three_d=1,contour=1)

# n = 100
# X = np.linspace(-20,20,n) #0.5 20*20
# Y = np.linspace(-5,5,n) #0.1 20*20
# X,Y = np.meshgrid(X,Y)
# Z = np.zeros((n,n))

# for i in range(n):
#   for j in range(n):
#     Z[j,i] = quad_value_func(np.array([X[0,i],Y[j,0]]),X_idare_2)

# C=plt.contour(X,Y,Z,levels=[10**i for i in range(5)], linestyles = 'dashed')
# plt.clabel(C, inline=True, fontsize=10, inline_spacing = 12)

# for i in range(n):
#   for j in range(n):
#     Z[j,i] = quad_value_func(np.array([X[0,i],Y[j,0]]),X_idare_1)

# C=plt.contour(X,Y,Z,levels=[10**i for i in range(5)])
# plt.clabel(C, inline=True, fontsize=10, inline_spacing = 12)
# plt.show()

# print(value_func(np.array([-20,0]),x_t,alphas,N_data))