"""
Finite-difference reference solver for 2D lid-driven cavity.
Stream function-vorticity formulation.

∇²ψ = -ω
∂ω/∂t + u·ω_x + v·ω_y = (1/Re)·∇²ω   (pseudo-time to steady state)
u = ∂ψ/∂y,  v = -∂ψ/∂x

Boundary conditions:
  ψ = 0  on all walls (stream function constant)
  Lid (y=1):  u = u_lid(x),  v = 0
  Other walls: u = v = 0
"""

import time
import numpy as np
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import factorized


def solve_cavity_fd(Re: float, n: int = 65, smooth_lid: bool = True,
                    max_iter: int = 25000, tol: float = 1e-7,
                    verbose: bool = True) -> dict:
    """
    Returns dict: x, y (1-D coords), u, v (n×n velocity), psi, omega,
                  iters, elapsed.
    Array indexing: arr[ix, iy]  with ix along x, iy along y.
    """
    t0 = time.time()
    h = 1.0 / (n - 1)
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)

    # Lid velocity — smooth profile removes corner singularity
    u_lid = 16.0 * x**2 * (1.0 - x)**2 if smooth_lid else np.ones(n)

    m = n - 2  # interior grid size per dimension

    # ── Build and factorize Laplacian for ψ (done once) ──────────
    # 5-point stencil: (ψ[i+1,j]+ψ[i-1,j]+ψ[i,j+1]+ψ[i,j-1]-4ψ[i,j])/h² = -ω[i,j]
    e  = np.ones(m)
    T1 = diags([e[:-1], -2.0*e, e[:-1]], [-1, 0, 1], format='csr')
    Im = eye(m, format='csr')
    L  = (1.0/h**2) * (kron(Im, T1, format='csr') + kron(T1, Im, format='csr'))
    solve_psi = factorized(L)

    # ── Initialise ───────────────────────────────────────────────
    psi   = np.zeros((n, n))
    omega = np.zeros((n, n))

    # CFL-limited time step (advection + diffusion)
    dt = 0.45 * min(h, 0.25 * h**2 * Re)

    # ── Pseudo-time loop ─────────────────────────────────────────
    for it in range(1, max_iter + 1):

        # Velocities from stream function (central differences, interior)
        u = np.zeros((n, n))
        v = np.zeros((n, n))
        u[1:-1, 1:-1] = (psi[1:-1, 2:] - psi[1:-1, :-2]) / (2*h)   # ∂ψ/∂y
        v[1:-1, 1:-1] = -(psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2*h)  # -∂ψ/∂x
        u[1:-1, -1]   = u_lid[1:-1]   # top lid BC

        # Interior velocity slices
        ui = u[1:-1, 1:-1]
        vi = v[1:-1, 1:-1]
        om = omega

        # Upwind advection
        adv_x = (np.maximum(ui, 0) * (om[1:-1,1:-1] - om[:-2, 1:-1])
               + np.minimum(ui, 0) * (om[2:,  1:-1] - om[1:-1, 1:-1])) / h
        adv_y = (np.maximum(vi, 0) * (om[1:-1,1:-1] - om[1:-1, :-2])
               + np.minimum(vi, 0) * (om[1:-1,2:]   - om[1:-1, 1:-1])) / h

        # Central diffusion
        lap = (om[2:,1:-1] + om[:-2,1:-1] + om[1:-1,2:] + om[1:-1,:-2]
               - 4*om[1:-1,1:-1]) / h**2

        # Time advance interior ω
        omega_new = omega.copy()
        omega_new[1:-1, 1:-1] = om[1:-1,1:-1] + dt*(-adv_x - adv_y + lap/Re)

        # Jensen boundary vorticity (Thom's formula)
        omega_new[1:-1,  0] = -2*psi[1:-1,  1] / h**2                    # bottom
        omega_new[1:-1, -1] = -2*psi[1:-1, -2] / h**2 - 2*u_lid[1:-1]/h # top
        omega_new[ 0, 1:-1] = -2*psi[ 1,  1:-1] / h**2                   # left
        omega_new[-1, 1:-1] = -2*psi[-2,  1:-1] / h**2                   # right

        # Solve Poisson: L ψ_int = -ω_int
        rhs = -omega_new[1:-1, 1:-1].flatten(order='C')
        psi[1:-1, 1:-1] = solve_psi(rhs).reshape(m, m, order='C')

        # Convergence
        delta = float(np.max(np.abs(omega_new - omega)))
        omega[:] = omega_new

        if verbose and it % 5000 == 0:
            print(f"    FD Re={Re:.0f}  it={it:5d}  |Δω|={delta:.2e}  "
                  f"({time.time()-t0:.0f}s elapsed)")
        if delta < tol:
            break

    elapsed = time.time() - t0
    if verbose:
        print(f"    FD Re={Re:.0f}  → {it} iters, |Δω|={delta:.2e}, {elapsed:.1f}s")

    return dict(x=x, y=y, u=u, v=v, psi=psi, omega=omega,
                iters=it, elapsed=elapsed, converged=(delta < tol))


def centreline_u(fd_result: dict) -> tuple:
    """
    Extract u-velocity on vertical centreline x=0.5 from FD result.
    Returns (y, u) arrays of length n.
    """
    result = fd_result
    x = result['x']
    n = len(x)
    ix_mid = n // 2    # index closest to x=0.5
    return result['y'], result['u'][ix_mid, :]
