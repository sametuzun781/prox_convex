import numpy as np

# n = 20*2*10 # dimension of x
# m = 50*2*10 # number of g_i terms

# # n = 20*2 # dimension of x
# # m = 50*2 # number of g_i terms

# # n = 20 # dimension of x
# # m = 50 # number of g_i terms

# # n = 2 # dimension of x
# # m = 5 # number of g_i terms

# # ------------------------------------------------------------

# A = np.load('A.npy')
# b = np.load('b.npy')

# # A = np.random.randn(m, n),
# # b = np.random.randn(m),

# # with open('A.npy', 'wb') as f:
# #     np.save(f, A)

# # with open('b.npy', 'wb') as f:
# #     np.save(f, b)

# # ------------------------------------------------------------

# epsilon_val = 1e-2

# # ------------------------------------------------------------



# def generate_random_positive_definite_matrix(n):
#     random_matrix = np.random.rand(n, n)
#     covariance_matrix = np.dot(random_matrix, random_matrix.T)
#     epsilon = 1e-6  # A small positive value to ensure positive definiteness
#     positive_definite_matrix = covariance_matrix + epsilon * np.identity(n)
#     diagonal_elements = np.sqrt(np.diag(positive_definite_matrix))
#     normalized_matrix = positive_definite_matrix / np.outer(diagonal_elements, diagonal_elements)

#     return normalized_matrix

# m = 2
# n = 2

# asas = 1
# asaf = 10

# A = [generate_random_positive_definite_matrix(n) for _ in range(m)]
# # A = [np.array([[1000, 0],[0, 1]]), np.array([[2, 0],[0, 1]])]
# a = [asas * (np.random.rand(n) - 0.5) for _ in range(m)]
# c = [10 + asaf * np.random.rand(1) for _ in range(m)]








m = 11
n = 2*m

s = 1.0

# A = [generate_random_positive_definite_matrix(n) for _ in range(m)]
A = [np.zeros((n,n)) for _ in range(m)]
for i in range(m):
    A[i][2*i, 2*i] = 1
    A[i][2*i+1,2*i+1] = s

a = [np.zeros(n) for _ in range(m)]
c = [0.0 for _ in range(m)]


dist_btw_pnts = 4.01
# x_init = [15.0, -20.0]
# x_final = [25.0, 20.0]
x_init = [25.0, -5.0]
x_final = [25.0, 35.0]
# tr_rad = 1e2





w_ptr_val = 1e-1
term_val = 1e-4
ITE = 500

# x_val = np.zeros(n)
# x_val = np.array([2., 1.])
# x_val = np.array([1., 2.])
# x_val = np.array([20., 20., 15., 7.])

x_val = np.zeros((m, 2))
x_val[:, 0] = np.linspace(x_init[0], x_final[0], m)
x_val[:, 1] = np.linspace(x_init[-1], x_final[-1], m)
x_val = x_val.flatten()

y_val = np.zeros(m)

solver_list = ['OSQP', 'QOCO', 'MOSEK', 'CLARABEL', 'ECOS'][1]

use_generated_code = False

# ------------------------------------------------------------
