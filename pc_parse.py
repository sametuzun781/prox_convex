import os
import cvxpy as cp
from cvxpygen import cpg

def parse_problem(alg_type, prb_params):
    # CVXPY variables and parameters
    x = cp.Variable(prb_params['x_dim'], name='x')
    dx = cp.Variable(prb_params['x_dim'], name='dx')

    xk = cp.Parameter(prb_params['x_dim'], name='xk')
    w_ptr = cp.Parameter(nonneg=True, name='w_ptr')

    if alg_type == 'prox_gradient':
        sR = cp.Parameter(name='sR') 
        grad_sR = cp.Parameter(prb_params['x_dim'], name='grad_sR')

        hC = cp.Parameter(name='hC') 
        grad_hC = cp.Parameter(prb_params['x_dim'], name='grad_hC')

        # Define the linearized function L(x_{k+1}, x_k)
        obj = cp.Minimize(prb_params['convex_cp'](x) + 
                          sR + grad_sR @ dx + 
                          hC + grad_hC @ dx + 
                          w_ptr * cp.sum_squares(dx))
        
    elif alg_type == 'prox_linear':
        sR = cp.Parameter(name='sR') 
        grad_sR = cp.Parameter(prb_params['x_dim'], name='grad_sR')

        C = cp.Parameter(prb_params['C_dim'], name='C')
        g_C = cp.Parameter((prb_params['C_dim'], prb_params['x_dim']), name='g_C')

        obj = cp.Minimize(prb_params['convex_cp'](x) + 
                          sR + grad_sR @ dx +
                          prb_params['h_cp'](C + g_C @ dx) +
                          w_ptr * cp.sum_squares(dx))

    elif alg_type == 'prox_convex':
        sR = cp.Parameter(name='sR') 
        grad_s = cp.Parameter(prb_params['R_dim'], nonneg=True, name='grad_s')
        grad_s_Rk = cp.Parameter(name='grad_s_Rk')

        R = prb_params['R_cp'](x)
        C = cp.Parameter(prb_params['C_dim'], name='C')
        g_C = cp.Parameter((prb_params['C_dim'], prb_params['x_dim']), name='g_C')

        obj = cp.Minimize(prb_params['convex_cp'](x) + 
                          sR + grad_s @ R - grad_s_Rk + 
                          prb_params['h_cp'](C + g_C @ dx) + 
                          w_ptr * cp.sum_squares(dx))

    # Define the linearized convex subproblem
    cons = [x == xk + dx,
            ]
    prb = cp.Problem(obj, cons)

    print('Problem parameters: ', prb.param_dict.keys())

    print('DPP: ', obj.is_dcp(dpp=True))

    return prb

def generate_problem(alg_type, prb_params):
    prb_ = parse_problem(alg_type, prb_params)
    code_dir = 'solver_' + alg_type
    # !rm -rf solver_prox_gradient/*
    os.makedirs(code_dir, exist_ok=True)
    cpg.generate_code(prb_, solver = "QOCO", wrapper = True, code_dir=code_dir)
