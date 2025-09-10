import time
import pickle
import numpy as np

# from pc_glob import term_val, ITE, use_generated_code, solver_list
from pc_glob import *

from pc_comp_params import compute_params_lin, compute_params_cvx
# from pc_parse import parse_linear_problem, parse_convex_problem
from pc_parse import parse_problem

from pc_fcns import convex_cp_2

def prox_linear(x_val, w_ptr_val, fcns_jit, prb_type):

    # ------------------------------------------

    if use_generated_code:
        if prb_type == 'lin':
            with open('solver_lin/problem.pickle', 'rb') as f:
                prb = pickle.load(f)

            from solver_lin.cpg_solver import cpg_solve

        elif prb_type == 'cvx':
            with open('solver_cvx/problem.pickle', 'rb') as f:
                prb = pickle.load(f)       
                
            from solver_cvx.cpg_solver import cpg_solve
        
        prb.register_solve('CPG', cpg_solve)

    else:
        prb = parse_problem(prb_type)

    # ------------------------------------------

    dx = prb.var_dict["dx"]
    x = prb.var_dict["x"]

    xk = prb.param_dict["xk"]
    w_ptr = prb.param_dict["w_ptr"]

    hg = prb.param_dict["hg"]
    if prb_type == 'lin':
        grad_hg = prb.param_dict["grad_hg"]
    elif prb_type == 'cvx':
        grad_h = prb.param_dict["grad_h"]
        grad_h_gk = prb.param_dict["grad_h_gk"]

    # ------------------------------------------

    w_ptr.value = w_ptr_val
    xk.value = x_val
    J_xk = fcns_jit['comp_jit'](x_val) + convex_cp_2(x_val)

    x_val_list = [x_val]
    J_list = [J_xk]
    dx_list = []
    times_cvx = []

    # ------------------------------------------

    converged = False
    J_max = 0
    t0 = time.time()
    for it in range(ITE):

        if converged:
            break
        
        # Set parameters
        if prb_type == 'lin':
            hg.value, grad_hg.value = compute_params_lin(xk.value, fcns_jit)
            a1, a2, a3 = compute_params_cvx(xk.value, fcns_jit)
            # print(np.max(grad_hg.value), np.max(a2))
            # if J_max < np.max(grad_hg.value):
            #     J_max = np.max(grad_hg.value)
            #     print('New J_max : ', J_max)
        elif prb_type == 'cvx':
            hg.value, grad_h.value, grad_h_gk.value = compute_params_cvx(xk.value, fcns_jit)

        while True:
            t00 = time.time()
            try:
                prb.solve(solver=solver_list, verbose=False)
            except:
                print('Solver failed, trying ECOS...')
                try:
                    prb.solve(solver='ECOS', verbose=False)
                except:
                    print('ECOS failed, trying MOSEK...')
                    prb.solve(solver='MOSEK', verbose=False)

            times_cvx.append(time.time() - t00)

            # Compute fcns_jit['comp_jit'](x.value) = h(g(x.value)) and fcns_jit['comp_jit'](x_{k+1}) = h(g(x_{k+1}))
            J_xk1 = fcns_jit['comp_jit'](x.value) + convex_cp_2(x.value)

            J_diff = J_xk - J_xk1
            L_diff = J_xk - prb.value
            ratio = J_diff / L_diff

            # Print iteration-wise differences
            print(f"Iter {it}, J_xk: {J_xk:.4f}, J_diff: {J_diff:.4f}, L_diff: {L_diff:.4f}, Ratio: {ratio:.4f}, w_ptr: {w_ptr.value:.4f}, dt: {time.time() - t00:.4f}")

            if it != 0 and L_diff < -1e-6:
                print('L_diff < -1e-6', L_diff)
                w_ptr.value = w_ptr.value * 2
                break

            if L_diff < term_val or w_ptr.value > 1e8:
                print('L_diff < term_val, stopping...', L_diff, w_ptr.value)
                converged = True
                break

            if ratio < 0:
                print('-'*35 + 'Reject' + '-'*35)
                w_ptr.value = w_ptr.value * 2

            else:

                # Update parameters
                xk.value = x.value
                J_xk = J_xk1

                # Store values for the next iteration
                x_val_list.append(x.value)
                J_list.append(J_xk1)
                dx_list.append(np.linalg.norm(dx.value))

                if ratio < 0.1:
                    w_ptr.value = w_ptr.value * 2
                if ratio > 0.8:
                    w_ptr.value = w_ptr.value / 2

                break

    dt = time.time() - t0

    return dt, x_val_list, J_list, dx_list, times_cvx
