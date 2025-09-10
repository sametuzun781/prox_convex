import os
import cvxpy as cp
from cvxpygen import cpg

# from pc_glob import n, m, tr_rad, dist_btw_pnts, x_init, x_final
from pc_glob import *
from pc_fcns import convex_cp, convex_cp_2

def parse_problem(prb_type):
    # CVXPY variables and parameters
    x = cp.Variable(n, name='x')
    dx = cp.Variable(n, name='dx')

    xk = cp.Parameter(n, name='xk')
    w_ptr = cp.Parameter(nonneg=True, name='w_ptr')

    hg = cp.Parameter(name='hg') 

    if prb_type == 'lin':
        grad_hg = cp.Parameter(n, name='grad_hg')

        # Define the linearized function L(x_{k+1}, x_k) = hg + grad_hg @ (x_{k+1} - x_k) + w_ptr * cp.sum_squares(x_{k+1} - x_k)
        obj = cp.Minimize(convex_cp_2(x) + hg + grad_hg @ dx + w_ptr * cp.sum_squares(dx))
        # obj = cp.Minimize(hg + grad_hg @ dx + w_ptr * cp.sum_squares(dx))
    elif prb_type == 'cvx':
        grad_h = cp.Parameter(m, nonneg=True, name='grad_h')
        grad_h_gk = cp.Parameter(name='grad_h_gk')

        g = convex_cp(x)
        obj = cp.Minimize(convex_cp_2(x) + hg + grad_h @ g - grad_h_gk + w_ptr * cp.sum_squares(dx))
        # obj = cp.Minimize(hg + grad_h @ g - grad_h_gk + w_ptr * cp.sum_squares(dx))

    # Define the linearized convex subproblem
    cons = [x == xk + dx,
                # cp.norm(dx) <= tr_rad,
                ]

    cons += [x[0:2] == x_init,
            #  x[-2:] == x_final,
            ]

    cons += [(x[2*i] - x[2*i + 2])**2 + (x[2*i+1] - x[2*i + 3])**2 <= dist_btw_pnts**2 for i in range(m-1)]
    
    prb = cp.Problem(obj, cons)

    print('DPP: ', obj.is_dcp(dpp=True))

    return prb

def generate_lin_problem():
    prb_lin = parse_problem(prb_type='lin')
    code_dir = "solver_lin"
    # !rm -rf solver_lin/*
    os.makedirs(code_dir, exist_ok=True)
    cpg.generate_code(prb_lin, solver = "QOCO", wrapper = True, code_dir=code_dir)

def generate_cvx_problem():
    prb_cvx = parse_problem(prb_type='cvx')
    code_dir = "solver_cvx"
    # !rm -rf solver_cvx/*
    os.makedirs(code_dir, exist_ok=True)
    cpg.generate_code(prb_cvx, solver = "QOCO", wrapper = True, code_dir=code_dir)
