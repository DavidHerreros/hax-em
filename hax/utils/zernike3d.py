# **************************************************************************
# *
# * Authors:  David Herreros Calero (dherreros@cnb.csic.es)
# *
# * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************


import jax
from einops.array_api import rearrange
from jax import numpy as jnp
from jax.scipy.special import factorial
import numpy as np
import sympy as sp


def precompute_legendre_Q(L2_MAX):
    x = sp.symbols('x')
    max_deg = L2_MAX

    # will fill into a numpy array first
    legQ = np.zeros((L2_MAX+1, L2_MAX+1, max_deg+1), dtype=np.float64)

    for l2 in range(L2_MAX+1):
        for m in range(l2+1):
            # 1) full assoc. Legendre
            Pexpr = sp.assoc_legendre(l2, m, x).expand()

            # 2) divide off (1 - x^2)^(m/2)
            Qexpr = sp.simplify(Pexpr / (1 - x ** 2) ** (sp.Rational(m, 2)))

            # 3) force it into a rational polynomial
            Qpoly = sp.Poly(Qexpr, x)
            coeffs = Qpoly.all_coeffs()    # list: [a_d, a_{d-1}, …, a_0]

            # 4) pack into fixed-length, highest-first array
            arr = np.array(coeffs, dtype=np.float64)
            pad = max_deg + 1 - arr.shape[0]
            arr_padded = np.pad(arr, (pad, 0), mode='constant')

            legQ[l2, m, :] = arr_padded

    # convert to JAX array
    return jnp.array(legQ)

def precompute_chebyshev_TU(M_MAX):
    x = sp.symbols('x')
    max_deg = M_MAX
    Tcoefs = np.zeros((M_MAX+1, max_deg+1), np.float64)
    Ucoefs = np.zeros((M_MAX+1, max_deg+1), np.float64)

    for m in range(M_MAX+1):
        Tpoly = sp.Poly(sp.chebyshevt(m, x).expand(), x, domain='QQ')
        Tp    = np.array(Tpoly.all_coeffs(), np.float64)
        padT  = max_deg+1 - Tp.shape[0]
        Tcoefs[m,:] = np.pad(Tp, (padT,0))

        Upoly = sp.Poly(sp.chebyshevu(m, x).expand(), x, domain='QQ')
        Up    = np.array(Upoly.all_coeffs(), np.float64)
        padU  = max_deg+1 - Up.shape[0]
        Ucoefs[m,:] = np.pad(Up, (padU,0))

    return jnp.array(Tcoefs), jnp.array(Ucoefs)


def precomputePolynomialsSph(l2_max):
    legendre_Q = precompute_legendre_Q(l2_max)
    T_coefs, U_coefs = precompute_chebyshev_TU(l2_max)
    return legendre_Q, T_coefs, U_coefs


def precomputePolynomialsZernike(l_max, n_max):
    # 1) prepare Sympy
    r = sp.symbols('r')
    radial_np = np.zeros((n_max+1, l_max+1, n_max+1), dtype=np.float64)

    # 2) for each (n,l) with same parity
    for n in range(n_max+1):
        for l in range(min(l_max, n)+1):
            if (n - l) % 2 != 0:
                continue
            k = (n - l) // 2

            # 2a) build the raw (unnormalized) radial expr
            #    Jacobi P_k^(alpha, l+1/2) at x = 2r^2 - 1
            Pk = sp.jacobi(k, 0, l + sp.Rational(1,2), sp.Symbol('x'))
            expr = ((-1) ** n) * (r ** l) * Pk.subs('x', 2 * r ** 2 - 1)

            # 2b) expand to a polynomial in r
            poly = sp.Poly(sp.expand(expr), r, domain='QQ')
            coeffs = np.array(poly.all_coeffs(), dtype=np.float64)  # highest-first

            # 2c) pad to length n_max+1
            pad = n_max + 1 - coeffs.shape[0]
            coeffs_padded = np.pad(coeffs, (pad, 0), mode='constant')

            radial_np[n, l, :] = coeffs_padded

    return jnp.array(radial_np)


def computeZernikes3D(l1, n, l2, m, pos, r_max, legendre_Q_coefs, T_coefs, U_coefs, zernike_coeffs):

    # General variables
    pos_r = pos / r_max
    xr, yr, zr = pos_r[..., 0], pos_r[..., 1], pos_r[..., 2]

    # Zernike Polynomials
    eps = 1e-6

    r2_safe = xr * xr + yr * yr + zr * zr + eps * eps
    r_safe = jnp.sqrt(r2_safe)

    # Table is indexed [total degree, angular degree]
    R_coefs = zernike_coeffs[l1.astype(jnp.int32), n.astype(jnp.int32)]
    R_vals = jnp.polyval(R_coefs, r_safe)
    R = jnp.sqrt(2. * l1 + 3.) * R_vals

    # Spherical Harmonics
    eps = 1e-6

    r2_safe = xr * xr + yr * yr + zr * zr + eps * eps
    rho2_xy = xr * xr + yr * yr + eps * eps
    r_safe  = jnp.sqrt(r2_safe)
    rho_xy = jnp.sqrt(rho2_xy)
    cos_ph = zr / r_safe
    sin_ph = rho_xy / r_safe
    cos_th = xr / rho_xy
    sin_th = yr / rho_xy
    abs_m = jnp.abs(m)

    # --- associated‐Legendre: P_ℓ2^|m|(cosφ) = (sinφ)^m Q(cosφ) ---
    # pick Q‐coefs by dynamic indexing
    Q_coefs = legendre_Q_coefs[l2.astype(jnp.int32), abs_m.astype(jnp.int32)]  # shape (13,)
    Q_vals = jnp.polyval(Q_coefs, cos_ph)  # shape (N,)
    P_vals = (sin_ph ** abs_m) * Q_vals

    # normalization
    norm = jnp.sqrt((2. * l2 + 1.) / (4. * jnp.pi)
                    * factorial(l2 - abs_m) / factorial(l2 + abs_m))
    norm = norm * jnp.where(m == 0, 1.0, jnp.sqrt(2.0))

    # --- azimuthal Chebyshev for cos(mθ), sin(mθ) ---
    Tm_coefs = T_coefs[abs_m.astype(jnp.int32)]  # shape (13,)
    cos_mth = jnp.polyval(Tm_coefs, cos_th)

    Um_idx = jnp.maximum(abs_m - 1., 0)
    Um_coefs = U_coefs[Um_idx.astype(jnp.int32)]  # shape (13,)
    sin_mth = sin_th * jnp.polyval(Um_coefs, cos_th)

    Y_az = jnp.where(m >= 0, cos_mth, sin_mth)
    Y = norm * P_vals * Y_az

    # Make zero those positions where d_pos_r > 1
    Z = R * Y
    outside = jnp.linalg.norm(pos_r, axis=-1) > 1.0
    Z = jnp.where(outside, 0.0, Z)

    return Z

def computeBasis(pos, r, sph_coeffs, zernike_coeffs, L1=None, L2=None, degrees=None, groups=None, centers=None):
    if degrees is None:
        degrees = basisDegreeVectors(L1, L2)

    if centers is None:
        def computeWithVMAP(degree):
            return computeZernikes3D(degree[0], degree[1], degree[2], degree[3], pos, r, sph_coeffs[0], sph_coeffs[1], sph_coeffs[2], zernike_coeffs)
        basis = jax.vmap(computeWithVMAP)(degrees)

        # FIXME: FASTEST BUT MORE COMPILATION TIME
        # basis = [computeZernikes3D(degrees[idx][0], degrees[idx][1], degrees[idx][2], degrees[idx][3],
        #          pos, r) for idx in range(len(degrees))]
        # basis = jnp.stack(basis, axis=0)

        # def computeWithVMAP(degree):
        #     return computeZernikes3D(degree[0], degree[1], degree[2], degree[3], pos, r)
        # vmapped_fn_for_chunks = jax.vmap(computeWithVMAP)
        # basis_chunks = []
        # for i in range(0, len(degrees), 10):
        #     current_degrees_chunk = degrees[i: i + 10]
        #     chunk_result = vmapped_fn_for_chunks(current_degrees_chunk)
        #     basis_chunks.append(chunk_result)
        # basis = jnp.concatenate(basis_chunks, axis=0)

        # FIXME: TWICE AS SLOW AS FASTEST BUT LESS COMPILATION TIME
        # def computeWithMAP(degree):
        #     return computeZernikes3D(degree[0], degree[1], degree[2], degree[3], pos, r, sph_coeffs[0], sph_coeffs[1], sph_coeffs[2], zernike_coeffs)
        # basis = jlx.map(computeWithMAP, degrees, batch_size=None)
    else:
        def computeWithVMAP(degree):
            return computeZernikes3D(degree[0], degree[1], degree[2], degree[3], centers, r, sph_coeffs[0], sph_coeffs[1], sph_coeffs[2], zernike_coeffs)
        basis_centers = jax.vmap(computeWithVMAP)(degrees)
        basis = jnp.zeros((pos.shape[0], basis_centers.shape[0]))
        basis_centers = rearrange(basis_centers, "c B -> B c")
        for group, basis_center in zip(groups[1], basis_centers):
            basis = basis.at[jnp.where(groups[0] == group, 1, 0)].set(basis_center)
        basis = rearrange(basis, "c B -> B c")

    return basis

def basisDegreeVectors(L1, L2):
    degrees = []

    # Compute basis degrees for each component
    for h in range(0, L2 + 1):
        totalSPH = 2 * h + 1
        aux = np.floor(totalSPH / 2)
        for l in range(h, L1 + 1, 2):
            for m in range(totalSPH):
                degrees.append([l, h, h, m - aux])

    return jnp.array(degrees)
