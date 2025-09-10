import numpy as np
import matplotlib.pyplot as plt

def plot(J_list_lin, J_list_cvx, x_val_list_lin, x_val_list_cvx, fcns_jit):

    plt.figure(figsize=(6, 3))
    # plt.plot(np.log(J_list_lin))
    # plt.plot(np.log(J_list_cvx))
    plt.plot(J_list_lin)
    plt.plot(J_list_cvx)

    try:
        x_vals = np.array(x_val_list_lin)
        x_vals_cvx = np.array(x_val_list_cvx)

        # Create a grid for plotting level sets
        x1 = np.linspace(-2, 2, 200)
        x2 = np.linspace(-2, 2, 200)
        X1, X2 = np.meshgrid(x1, x2)
        Z = np.zeros_like(X1)

        # Evaluate comp_jit(x) on the grid
        for i in range(X1.shape[0]):
            for j in range(X1.shape[1]):
                x_ij = np.array([X1[i, j], X2[i, j]])
                Z[i, j] = fcns_jit['comp_jit'](x_ij)

        # Plot level sets and SCP trajectory
        plt.figure(figsize=(8, 6))
        CS = plt.contour(X1, X2, Z, levels=30, cmap='viridis')
        plt.clabel(CS, inline=1, fontsize=8)
        plt.plot(x_vals[:, 0], x_vals[:, 1], 'r.-', linewidth=2, label="SCP trajectory")
        # plt.scatter(x_vals[0, 0], x_vals[0, 1], c='blue', label='Start', s=60)
        # plt.scatter(x_vals[-1, 0], x_vals[-1, 1], c='red', label='End', s=60)
        plt.plot(x_vals_cvx[:, 0], x_vals_cvx[:, 1], c='blue', ls='-', linewidth=1, label="SCP trajectory cvx")
        plt.title("SCP trajectory over level sets of comp_jit(x) = h(g(x))")
        # plt.xlim(-1.0, 1.5)
        # plt.ylim(-1.1, 1.9)
        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")
        plt.legend()
        plt.grid(True)
        # plt.axis('equal')
        plt.show()
    except:
        pass
