import cvxpy as cp
import jax.numpy as jnp
from jax import vmap, config
config.update("jax_enable_x64", True)











# from pc_glob import A, b, epsilon_val

# def convex_cp(x_val):
#     return cp.square(A @ x_val - b)

# def convex_jax(x_val):
#     return jnp.square(A @ x_val - b)

# def smooth(y_val):
#     return jnp.sum(jnp.log(1 + y_val / epsilon_val))

# def comp(x_val):
#     return smooth(convex_jax(x_val))





# from pc_glob import *

# def convex_cp(x_val):
#     terms = []
#     for i in range(m):
#         # Ensure symmetry if needed: A = 0.5*(A + A.T)
#         quad = 0.5 * cp.quad_form(x_val, A[i])
#         lin  = a[i] @ x_val         # cp.matmul(a, x_val) also fine
#         terms.append(quad + lin +  c[i])
#     return 0.1 * cp.hstack(terms)



# As = jnp.stack(A, axis=0)   # (8, n, n)
# as_ = jnp.stack(a, axis=0)  # (8, n)
# cs = jnp.array(c)           # (8,)

# def _quad_affine(A, a, c, x):
#     # 0.5 * x^T A x + a^T x + c
#     quad = 0.5 * (x @ A @ x)
#     lin  = a @ x
#     return quad + lin + c

# # Vectorize over the first dimension of (As, as_, cs)
# _batched = vmap(_quad_affine, in_axes=(0, 0, 0, None), out_axes=0)

# def convex_jax(x_val):
#     return 0.1 * _batched(As, as_, cs, x_val)[:,0]  # shape (8,)











from pc_glob import *


sc = 1/100
c_smth = 0.2
def convex_cp(x_val):
    terms = []
    for i in range(m):
        # Ensure symmetry if needed: A = 0.5*(A + A.T)
        quad = 0.5 * cp.quad_form(x_val, A[i])
        lin  = a[i] @ x_val         # cp.matmul(a, x_val) also fine
        terms.append(quad + lin +  c[i])
    return sc * cp.hstack(terms)

As = jnp.stack(A, axis=0)   # (8, n, n)
as_ = jnp.stack(a, axis=0)  # (8, n)
cs = jnp.array(c)           # (8,)

def _quad_affine(A, a, c, x):
    # 0.5 * x^T A x + a^T x + c
    quad = 0.5 * (x @ A @ x)
    lin  = a @ x
    return quad + lin + c

# Vectorize over the first dimension of (As, as_, cs)
_batched = vmap(_quad_affine, in_axes=(0, 0, 0, None), out_axes=0)

def convex_jax(x_val):
    # return _batched(As, as_, cs, x_val)[:,0]  # shape (8,)
    return sc * _batched(As, as_, cs, x_val)  # shape (8,)













def smooth(y_val):
    return (jnp.prod(y_val) + c_smth**m)**(1/m) - c_smth

def comp(x_val):
    return smooth(convex_jax(x_val))



def convex_cp_2(x):
    # z = 0.0
    # for i in range(m):
        # z += (-x[i])

    return -0.04*x[-1]
