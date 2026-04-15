import time
import pickle
import numpy as np

from pc_parse import parse_problem


def _get_param(prb, name):
    """Return a CVXPY parameter if it exists in the compiled problem, else None."""
    return prb.param_dict.get(name, None)


def _set_param(param, value):
    """Safely assign a value only when the parameter is present in the problem."""
    if param is not None:
        param.value = np.asarray(value)


def prox_convex(prb_params, alg_type):

    # ------------------------------------------

    if prb_params['use_generated_code']:
        if alg_type == 'prox_gradient':
            with open('solver_prox_gradient/problem.pickle', 'rb') as f:
                prb = pickle.load(f)
            from solver_prox_gradient.cpg_solver import cpg_solve

        elif alg_type == 'prox_linear':
            with open('solver_prox_linear/problem.pickle', 'rb') as f:
                prb = pickle.load(f)
            from solver_prox_linear.cpg_solver import cpg_solve

        elif alg_type == 'prox_convex':
            with open('solver_prox_convex/problem.pickle', 'rb') as f:
                prb = pickle.load(f)
            from solver_prox_convex.cpg_solver import cpg_solve

        else:
            raise ValueError(f"Unknown alg_type: {alg_type}")

        prb.register_solve('CPG', cpg_solve)

    else:
        prb = parse_problem(alg_type, prb_params)

    # ------------------------------------------

    dx = prb.var_dict["dx"]
    x = prb.var_dict["x"]

    xk = prb.param_dict["xk"]
    w_ptr = prb.param_dict["w_ptr"]

    # Optional parameters: CVXPY may prune inactive parameters from param_dict.
    sR = _get_param(prb, "sR")
    grad_sR = _get_param(prb, "grad_sR")

    hC = _get_param(prb, "hC")
    grad_hC = _get_param(prb, "grad_hC")

    C = _get_param(prb, "C")
    g_C = _get_param(prb, "g_C")

    grad_s = _get_param(prb, "grad_s")
    grad_s_Rk = _get_param(prb, "grad_s_Rk")

    # ------------------------------------------
    # Termination tolerances
    # ------------------------------------------
    term_val = prb_params.get('term_val', 1e-8)
    term_dx = prb_params.get('term_dx', 1e-10)

    # ------------------------------------------

    xk.value = np.asarray(prb_params['x_val'])
    w_ptr.value = float(prb_params['w_ptr_val'])
    J_xk = float(np.asarray(prb_params['comp'](prb_params['x_val'])))

    x_val_list = [np.asarray(xk.value).copy()]
    w_ptr_val_list = [float(w_ptr.value)]
    J_list = [J_xk]
    dx_list = []
    times_cvx = []

    # ------------------------------------------

    converged = False
    hit_max_iter = False
    t0 = time.time()
    for it in range(prb_params['ITE']):

        if converged:
            break

        # Set parameters for the requested algorithm.
        if alg_type == 'prox_gradient':
            _set_param(sR, prb_params['sR'](xk.value))
            _set_param(grad_sR, prb_params['g_sR'](xk.value))

            _set_param(hC, prb_params['hC'](xk.value))
            _set_param(grad_hC, prb_params['g_hC'](xk.value))

        elif alg_type == 'prox_linear':
            _set_param(sR, prb_params['sR'](xk.value))
            _set_param(grad_sR, prb_params['g_sR'](xk.value))

            _set_param(C, prb_params['C'](xk.value))
            _set_param(g_C, prb_params['g_C'](xk.value))

        elif alg_type == 'prox_convex':
            _set_param(sR, prb_params['sR'](xk.value))

            if (grad_s is not None) or (grad_s_Rk is not None):
                R_val = np.asarray(prb_params['R_jax'](xk.value))
                g_s_val = np.asarray(prb_params['g_s'](R_val))
                _set_param(grad_s, g_s_val)
                _set_param(grad_s_Rk, g_s_val @ R_val)

            _set_param(C, prb_params['C'](xk.value))
            _set_param(g_C, prb_params['g_C'](xk.value))

        else:
            raise ValueError(f"Unknown alg_type: {alg_type}")

        while True:
            t00 = time.time()
            try:
                prb.solve(solver=prb_params['solver_list'], verbose=False)
            except Exception:
                print('Solver failed, trying ECOS...')
                try:
                    prb.solve(solver='ECOS', verbose=False)
                except Exception:
                    print('ECOS failed, trying MOSEK...')
                    prb.solve(solver='MOSEK', verbose=False)

            times_cvx.append(time.time() - t00)

            J_xk1 = float(np.asarray(prb_params['comp'](x.value)))
            dx_norm = float(np.linalg.norm(dx.value))

            J_diff = J_xk - J_xk1
            L_diff = J_xk - float(prb.value)

            # Protect the ratio computation against divide-by-zero / tiny denominators.
            if abs(L_diff) <= 1e-12:
                ratio = np.inf if J_diff >= 0 else -np.inf
            else:
                ratio = J_diff / L_diff

            print(
                f"Iter {it}, J_xk: {J_xk:.4f}, J_diff: {J_diff:.6e}, "
                f"L_diff: {L_diff:.6e}, Ratio: {ratio:.4f}, "
                f"dx: {dx_norm:.6e}, w_ptr: {w_ptr.value:.4f}, "
                f"dt: {time.time() - t00:.4f}"
            )

            if not prb_params['adaptive_step']:
                xk.value = np.asarray(x.value)
                J_xk = J_xk1

                x_val_list.append(np.asarray(x.value).copy())
                w_ptr_val_list.append(float(w_ptr.value))
                J_list.append(J_xk1)
                dx_list.append(dx_norm)

                if (abs(J_diff) < term_val) or (dx_norm < term_dx) or (w_ptr.value >= 1e8):
                    converged = True

                break

            else:
                if L_diff < -1e-6:
                    print('Problem with the convex solver: L_diff < -1e-6', L_diff)
                    w_ptr.value = w_ptr.value * prb_params['r_inc']
                    break

                if ratio < prb_params['r0']:
                    print('-' * 35 + 'Reject' + '-' * 35)
                    w_ptr.value = w_ptr.value * prb_params['r_inc']

                    if w_ptr.value >= 1e8:
                        converged = True
                        break

                else:
                    xk.value = np.asarray(x.value)
                    J_xk = J_xk1

                    x_val_list.append(np.asarray(x.value).copy())
                    w_ptr_val_list.append(float(w_ptr.value))
                    J_list.append(J_xk1)
                    dx_list.append(dx_norm)

                    if ratio < prb_params['r1']:
                        w_ptr.value = w_ptr.value * prb_params['r_inc']
                    if ratio > prb_params['r2']:
                        w_ptr.value = w_ptr.value / prb_params['r_decr']

                    if (abs(J_diff) < term_val) or (dx_norm < term_dx) or (w_ptr.value >= 1e8):
                        converged = True

                    break

    scp_solve_time = time.time() - t0

    # Check if we exhausted iterations without converging
    if not converged:
        hit_max_iter = True

    return {
        "scp_solve_time": scp_solve_time,
        "x_val_list": x_val_list,
        "w_ptr_val_list": w_ptr_val_list,
        "J_list": J_list,
        "dx_list": dx_list,
        "times": times_cvx,
        "converged": converged,
        "hit_max_iter": hit_max_iter,
    }
