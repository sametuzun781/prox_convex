
import numpy as np

def compute_params_lin(x_val, fcns_jit):

    hg_val = np.asarray(fcns_jit['comp_jit'](x_val))
    grad_hg_val = np.asarray(fcns_jit['g_comp_jit'](x_val))

    return hg_val, grad_hg_val

def compute_params_cvx(x_val, fcns_jit):

    hg_val = np.asarray(fcns_jit['comp_jit'](x_val))

    y_val = fcns_jit['convex_jax_jit'](x_val)
    # print('y_val: ', np.max(y_val))
    grad_h_val = np.asarray(fcns_jit['g_smooth_jit'](y_val))
    # print('grad_h_val: ', grad_h_val)
    # print('grad_h_val: ', np.max(grad_h_val))
    grad_h_gk_val = grad_h_val @ np.asarray(y_val)
    # print('grad_h_gk_val: ', np.max(grad_h_gk_val))

    return hg_val, grad_h_val, grad_h_gk_val
