#!/usr/bin/env python3
"""Dependency-free experiments for the RRPE manuscript.

The final RRPE coefficient problem is solved as the stated nonsmooth convex
problem.  A first-order primal-dual routine is kept for reference, but it is
superseded by the active-set support/sign enumeration solver defined later in
this file.  Each reported solve records affine feasibility, objective value, and
a stationarity/KKT diagnostic.
"""
import itertools
import math
import random
import time

SOLVE_LOG = []


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm(v):
    return math.sqrt(max(0.0, dot(v, v)))


def add(a, b):
    return [x + y for x, y in zip(a, b)]


def sub(a, b):
    return [x - y for x, y in zip(a, b)]


def axpy(a, x, y):
    return [a * xi + yi for xi, yi in zip(x, y)]


def scal(a, x):
    return [a * xi for xi in x]


def lincomb(cols, alpha):
    n = len(cols[0])
    return [sum(alpha[j] * cols[j][i] for j in range(len(alpha))) for i in range(n)]


def rt_y(cols, y):
    return [dot(c, y) for c in cols]


def objective(cols, delta, alpha):
    return norm(lincomb(cols, alpha)) + sum(delta[j] * abs(alpha[j]) for j in range(len(alpha)))


def soft(x, t):
    if x > t:
        return x - t
    if x < -t:
        return x + t
    return 0.0


def prox_weighted_l1_affine(v, weights, target=1.0):
    def total(lam):
        return sum(soft(v[j] - lam, weights[j]) for j in range(len(v)))

    span = 1.0 + abs(target) + sum(abs(x) for x in v) + sum(weights)
    low = min(v[j] - weights[j] for j in range(len(v))) - span
    high = max(v[j] + weights[j] for j in range(len(v))) + span
    while total(low) < target:
        low = 2.0 * low - 1.0
    while total(high) > target:
        high = 2.0 * high + 1.0
    for _ in range(90):
        mid = 0.5 * (low + high)
        if total(mid) > target:
            low = mid
        else:
            high = mid
    lam = 0.5 * (low + high)
    return [soft(v[j] - lam, weights[j]) for j in range(len(v))]


def operator_norm(cols):
    p = len(cols)
    if p == 0:
        return 0.0
    v = [1.0 / math.sqrt(p)] * p
    for _ in range(80):
        kv = lincomb(cols, v)
        w = rt_y(cols, kv)
        nw = norm(w)
        if nw == 0:
            return 0.0
        v = [x / nw for x in w]
    return norm(lincomb(cols, v))


def best_stationarity_lambda(w, alpha, delta):
    p = len(alpha)
    lo = min(-w[j] - delta[j] - 1.0 for j in range(p))
    hi = max(-w[j] + delta[j] + 1.0 for j in range(p))

    def component(lam, j):
        u = w[j] + lam
        if abs(alpha[j]) > 1e-7:
            return u + delta[j] * (1.0 if alpha[j] > 0 else -1.0)
        if u > delta[j]:
            return u - delta[j]
        if u < -delta[j]:
            return u + delta[j]
        return 0.0

    def deriv(lam):
        return 2.0 * sum(component(lam, j) for j in range(p))

    guard = 0
    while deriv(lo) > 0 and guard < 50:
        lo -= 2.0 * (abs(lo) + 1.0)
        guard += 1
    guard = 0
    while deriv(hi) < 0 and guard < 50:
        hi += 2.0 * (abs(hi) + 1.0)
        guard += 1
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if deriv(mid) < 0:
            lo = mid
        else:
            hi = mid
    lam = 0.5 * (lo + hi)
    residual = norm([component(lam, j) for j in range(p)])
    return lam, residual


def rrpe_solve(cols, delta, starts=None, tol=1e-7, maxit=20000, name="rrpe"):
    p = len(delta)
    n = len(cols[0]) if cols else 0
    starts = starts or []
    start_list = [[1.0 / p] * p]
    for st in starts:
        if st is not None and len(st) == p:
            start_list.append(project_affine(st))
    L = operator_norm(cols)
    tau = sigma = 1.0 if L == 0 else 0.5 / L
    best = None
    best_info = None
    for start_id, a0 in enumerate(start_list):
        alpha = project_affine(a0)
        y = [0.0] * n
        previous = alpha[:]
        last_kkt = float("inf")
        it_done = maxit
        for it in range(1, maxit + 1):
            w = rt_y(cols, y)
            v = [alpha[j] - tau * w[j] for j in range(p)]
            anew = prox_weighted_l1_affine(v, [tau * d for d in delta])
            abar = [2.0 * anew[j] - alpha[j] for j in range(p)]
            ynew = add(y, scal(sigma, lincomb(cols, abar)))
            ny = norm(ynew)
            if ny > 1.0:
                ynew = [z / ny for z in ynew]
            alpha, y = anew, ynew
            if it % 100 == 0:
                _, last_kkt = best_stationarity_lambda(rt_y(cols, y), alpha, delta)
                step = norm([alpha[j] - previous[j] for j in range(p)])
                previous = alpha[:]
                if last_kkt <= tol and step <= 10.0 * tol:
                    it_done = it
                    break
        lam, kkt = best_stationarity_lambda(rt_y(cols, y), alpha, delta)
        info = {
            "name": name,
            "start": start_id,
            "iters": it_done,
            "feas": abs(sum(alpha) - 1.0),
            "objective": objective(cols, delta, alpha),
            "kkt": kkt,
            "dual_norm": norm(y),
            "lambda": lam,
        }
        if best is None or info["objective"] < best_info["objective"]:
            best = alpha
            best_info = info
    SOLVE_LOG.append(best_info)
    return best, best_info


def project_affine(a):
    correction = (sum(a) - 1.0) / len(a)
    return [x - correction for x in a]


def gaussian_elimination(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for k in range(n):
        pivot = max(range(k, n), key=lambda i: abs(M[i][k]))
        if abs(M[pivot][k]) < 1e-14:
            M[pivot][k] = 1e-14
        if pivot != k:
            M[k], M[pivot] = M[pivot], M[k]
        piv = M[k][k]
        for j in range(k, n + 1):
            M[k][j] /= piv
        for i in range(n):
            if i == k:
                continue
            factor = M[i][k]
            if factor == 0:
                continue
            for j in range(k, n + 1):
                M[i][j] -= factor * M[k][j]
    return [M[i][n] for i in range(n)]


def gram(cols):
    p = len(cols)
    return [[dot(cols[i], cols[j]) for j in range(p)] for i in range(p)]


def affine_quadratic_coeffs(cols, ridge=0.0):
    p = len(cols)
    G = gram(cols)
    K = [[0.0] * (p + 1) for _ in range(p + 1)]
    rhs = [0.0] * (p + 1)
    for i in range(p):
        for j in range(p):
            K[i][j] = G[i][j] + (ridge if i == j else 0.0)
        K[i][p] = 1.0
        K[p][i] = 1.0
    rhs[p] = 1.0
    try:
        sol = gaussian_elimination(K, rhs)
        return sol[:p]
    except Exception:
        if ridge == 0.0:
            return affine_quadratic_coeffs(cols, ridge=1e-10)
        return [1.0 / p] * p


def simplex_project(v):
    u = sorted(v, reverse=True)
    cssv = 0.0
    rho = 0
    theta = 0.0
    for i, ui in enumerate(u, 1):
        cssv += ui
        t = (cssv - 1.0) / i
        if ui - t > 0:
            rho = i
            theta = t
    return [max(x - theta, 0.0) for x in v]


def simplex_coeffs(cols, iters=1000):
    p = len(cols)
    G = gram(cols)
    L = max(operator_norm(cols) ** 2, 1e-12)
    a = [1.0 / p] * p
    for _ in range(iters):
        g = [2.0 * sum(G[i][j] * a[j] for j in range(p)) for i in range(p)]
        a = simplex_project([a[j] - g[j] / L for j in range(p)])
    return a


def ndtri(p):
    a = [-39.69683028665376, 220.9460984245205, -275.9285104469687,
         138.3577518672690, -30.66479806614716, 2.506628277459239]
    b = [-54.47609879822406, 161.5858368580409, -155.6989798598866,
         66.80131188771972, -13.28068155288572]
    c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838,
         -2.549732539343734, 4.374664141464968, 2.938163982698783]
    d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996,
         3.754408661907416]
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p <= phigh:
        q = p - 0.5
        r = q * q
        num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        den = (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        return num / den
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
    )


def coverage_tables():
    p = 6
    m = 16
    trials = 10000
    rng = random.Random(2)
    gaussian_samples = [[sum(rng.gauss(0.0, 1.0) for _ in range(m)) / m for _ in range(p)] for _ in range(trials)]
    out_gauss = []
    for nominal in [0.90, 0.95, 0.99]:
        u = (1.0 - nominal) / p
        radius = ndtri(1.0 - u / 2.0) / math.sqrt(m)
        col = sum(abs(x) <= radius for row in gaussian_samples for x in row) / (trials * p)
        win = sum(all(abs(x) <= radius for x in row) for row in gaussian_samples) / trials
        mean_err = sum(abs(x) for row in gaussian_samples for x in row) / (trials * p)
        out_gauss.append((nominal, radius, col, win, mean_err))

    # Concrete bounded-vector theoretical radius from the paper, using bounded
    # Rademacher samples with b=sigma=1 and nu=0.
    m2 = 32
    rng = random.Random(3)
    bounded_samples = [[sum(1.0 if rng.random() < 0.5 else -1.0 for _ in range(m2)) / m2 for _ in range(p)] for _ in range(trials)]
    out_pin = []
    for nominal in [0.90, 0.95, 0.99]:
        u = (1.0 - nominal) / p
        radius = 1.0 / math.sqrt(m2) + math.sqrt(2.0 * math.log(1.0 / u) / m2)
        col = sum(abs(x) <= radius for row in bounded_samples for x in row) / (trials * p)
        win = sum(all(abs(x) <= radius for x in row) for row in bounded_samples) / trials
        mean_err = sum(abs(x) for row in bounded_samples for x in row) / (trials * p)
        out_pin.append((nominal, radius, col, win, mean_err))
    return out_gauss, out_pin


def close_rows(p, h):
    lambdas = [1.0] + [1.0 - i * h for i in range(1, p)]
    return [[(lam - 1.0) * (lam ** j) for j in range(p)] for lam in lambdas]


def rows_to_cols(rows):
    return [[rows[i][j] for i in range(len(rows))] for j in range(len(rows[0]))]


def q_coeff(p, h):
    coeff = [1.0]
    for i in range(1, p):
        root = 1.0 - i * h
        new = [0.0] * (len(coeff) + 1)
        for j, c in enumerate(coeff):
            new[j] += -root * c / (i * h)
            new[j + 1] += c / (i * h)
        coeff = new
    return coeff


def phase_table():
    p = 6
    h = 0.2
    cols = rows_to_cols(close_rows(p, h))
    rre = q_coeff(p, h)
    prev = rre
    rows = []
    for delta in [1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]:
        alpha, info = rrpe_solve(cols, [delta] * p, starts=[prev, rre, affine_quadratic_coeffs(cols, 0.0)],
                                 tol=3e-7, maxit=12000, name="phase")
        prev = alpha
        pred = norm(lincomb(cols, alpha))
        beta = sum(abs(x) for x in alpha)
        j_rrpe = pred + delta * beta
        j_rre = norm(lincomb(cols, rre)) + delta * sum(abs(x) for x in rre)
        rows.append((delta, beta, pred, j_rrpe, j_rre, info["kkt"]))
    return rows


def scaling_table():
    p = 6
    hs = [0.1, 0.05, 0.02, 0.01, 0.005]
    rows = []
    for h in hs:
        beta = sum(abs(x) for x in q_coeff(p, h))
        lower = 1.0 / (math.factorial(p - 1) * h ** (p - 1))
        upper_const = beta * h ** (p - 1)
        rows.append((h, beta, lower, upper_const))
    xs = [math.log(r[0]) for r in rows]
    ys = [math.log(r[1]) for r in rows]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return rows, slope


def solve_poisson(rhs):
    n = len(rhs)
    h = 1.0 / (n + 1)
    lower = [-1.0 / (h * h)] * (n - 1)
    diag = [2.0 / (h * h)] * n
    upper = [-1.0 / (h * h)] * (n - 1)
    d = rhs[:]
    for i in range(1, n):
        w = lower[i - 1] / diag[i - 1]
        diag[i] -= w * upper[i - 1]
        d[i] -= w * d[i - 1]
    x = [0.0] * n
    x[-1] = d[-1] / diag[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - upper[i] * x[i + 1]) / diag[i]
    return x


def bratu_G(u, lam):
    return solve_poisson([lam * math.exp(ui) for ui in u])


def bratu_F(u, lam):
    return sub(bratu_G(u, lam), u)


def lambda_min_A(n):
    h = 1.0 / (n + 1)
    return 4.0 / (h * h) * math.sin(math.pi / (2.0 * (n + 1))) ** 2


def M_ub_bratu(vertices, lam, n):
    U = max(max(v) for v in vertices)
    return lam * math.exp(U) / lambda_min_A(n)


def D_defect(alpha, X, M):
    z = lincomb(X, alpha)
    return 0.5 * M * sum(abs(a) * dot(sub(x, z), sub(x, z)) for a, x in zip(alpha, X))


def quantile(vals, q):
    if not vals:
        return 0.0
    vals = sorted(vals)
    pos = q * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def run_bratu_method(method, n=24, lam=2.0, delta=1e-8, eta=0.95, memory=5, tol=1e-9, maxit=80):
    t0 = time.perf_counter()
    x = [0.0] * n
    X = []
    R = []
    cache_r = None
    total_ops = 0
    dedicated = 0
    attempts = cert = explicit = explicit_acc = fallback = 0
    max_alpha = 1.0
    ratios = []
    solve_kkt = []
    final_r = None
    for it in range(maxit + 1):
        if cache_r is None:
            r = bratu_F(x, lam)
            total_ops += 1
        else:
            r = cache_r
            cache_r = None
        rn = norm(r)
        final_r = rn
        if rn < tol or it == maxit:
            break
        X.append(x)
        R.append(r)
        if len(X) > memory:
            X = X[-memory:]
            R = R[-memory:]
        took = False
        if method != "Picard" and len(X) >= 2:
            attempts += 1
            p = len(X)
            if method == "RRE":
                alpha = affine_quadratic_coeffs(R, ridge=0.0)
                z = lincomb(X, alpha)
            elif method == "Ridge-RRE":
                alpha = affine_quadratic_coeffs(R, ridge=1e-6)
                z = lincomb(X, alpha)
            elif method == "Simplex-RRE":
                alpha = simplex_coeffs(R)
                z = lincomb(X, alpha)
            elif method == "Anderson":
                alpha = affine_quadratic_coeffs(R, ridge=1e-10)
                GX = [add(X[j], R[j]) for j in range(p)]
                z = lincomb(GX, alpha)
            elif method == "RRPE":
                rre_start = affine_quadratic_coeffs(R, ridge=1e-10)
                alpha, info = rrpe_solve(R, [delta] * p, starts=[rre_start], tol=5e-7, maxit=2500, name="bratu")
                solve_kkt.append(info["kkt"])
                z = lincomb(X, alpha)
                M = M_ub_bratu(X + [z], lam, n)
                A = norm(lincomb(R, alpha)) + delta * sum(abs(a) for a in alpha)
                B = A + D_defect(alpha, X, M)
                L = max(0.0, rn - delta)
                true_z_diag = bratu_F(z, lam)
                ratios.append(norm(true_z_diag) / B if B > 0 else 0.0)
                if B <= eta * L:
                    cert += 1
                    x = z
                    cache_r = None
                    took = True
                else:
                    explicit += 1
                    dedicated += 1
                    rz = true_z_diag
                    total_ops += 1
                    if norm(rz) <= eta * rn:
                        explicit_acc += 1
                        x = z
                        cache_r = rz
                        took = True
                    else:
                        fallback += 1
                        x = add(x, r)
                        cache_r = None
                        took = True
            else:
                raise ValueError(method)
            max_alpha = max(max_alpha, sum(abs(a) for a in alpha))
            if method != "RRPE":
                explicit += 1
                dedicated += 1
                rz = bratu_F(z, lam)
                total_ops += 1
                if norm(rz) <= eta * rn:
                    explicit_acc += 1
                    x = z
                    cache_r = rz
                    took = True
                else:
                    fallback += 1
                    x = add(x, r)
                    cache_r = None
                    took = True
        if not took:
            x = add(x, r)
            cache_r = None
    return {
        "method": method,
        "iters": it,
        "ops": total_ops,
        "dedicated": dedicated,
        "attempts": attempts,
        "cert": cert,
        "explicit": explicit,
        "explicit_acc": explicit_acc,
        "fallback": fallback,
        "max_alpha": max_alpha,
        "maxC": max(ratios) if ratios else 0.0,
        "medC": quantile(ratios, 0.5),
        "q95C": quantile(ratios, 0.95),
        "viol": sum(r > 1.0 + 1e-10 for r in ratios),
        "final": final_r,
        "time": time.perf_counter() - t0,
        "max_kkt": max(solve_kkt) if solve_kkt else 0.0,
    }


def bratu_benchmark():
    methods = ["Picard", "RRE", "Ridge-RRE", "Anderson", "Simplex-RRE", "RRPE"]
    return [run_bratu_method(m) for m in methods]


def matvec_A_tridiag(x):
    n = len(x)
    h = 1.0 / (n + 1)
    c = 1.0 / (h * h)
    y = [0.0] * n
    for i in range(n):
        val = 2.0 * c * x[i]
        if i > 0:
            val -= c * x[i - 1]
        if i + 1 < n:
            val -= c * x[i + 1]
        y[i] = val
    return y


def cg_poisson(rhs, reltol=1e-5, maxit=120, x0=None):
    n = len(rhs)
    x = [0.0] * n if x0 is None else x0[:]
    r = sub(rhs, matvec_A_tridiag(x))
    p = r[:]
    rr = dot(r, r)
    target = (reltol * max(norm(rhs), 1e-30)) ** 2
    if rr <= target:
        return x, math.sqrt(rr), 0
    for it in range(1, maxit + 1):
        Ap = matvec_A_tridiag(p)
        denom = dot(p, Ap)
        if denom <= 0:
            break
        alpha = rr / denom
        x = axpy(alpha, p, x)
        r = axpy(-alpha, Ap, r)
        rr_new = dot(r, r)
        if rr_new <= target:
            return x, math.sqrt(rr_new), it
        beta = rr_new / rr
        p = add(r, scal(beta, p))
        rr = rr_new
    return x, math.sqrt(rr), maxit


def bratu_G_inexact(u, lam, reltol):
    rhs = [lam * math.exp(ui) for ui in u]
    y, inner_res, inner_it = cg_poisson(rhs, reltol=reltol, maxit=120)
    return y, inner_res, inner_it


def bratu_F_true(u, lam):
    return bratu_F(u, lam)


def run_large_inexact(method, n=1000, lam=1.0, reltol=1e-3, memory=4, eta=0.95, maxit=12, tol=1e-7):
    t0 = time.perf_counter()
    x = [0.0] * n
    X = []
    R = []
    D = []
    attempts = cert = explicit = explicit_acc = fallback = 0
    total_ops = dedicated = inner_total = 0
    max_alpha = 1.0
    final_true = None
    solve_kkt = []
    lmin = lambda_min_A(n)
    for it in range(maxit + 1):
        gx, inner_res, inner_it = bratu_G_inexact(x, lam, reltol)
        total_ops += 1
        inner_total += inner_it
        r = sub(gx, x)
        delta_cur = inner_res / lmin
        true_r = bratu_F_true(x, lam)
        final_true = norm(true_r)
        if final_true < tol or it == maxit:
            break
        X.append(x)
        R.append(r)
        D.append(delta_cur)
        if len(X) > memory:
            X = X[-memory:]
            R = R[-memory:]
            D = D[-memory:]
        took = False
        if method != "Picard" and len(X) >= 2:
            attempts += 1
            if method == "Ridge-RRE":
                alpha = affine_quadratic_coeffs(R, ridge=1e-6)
                z = lincomb(X, alpha)
            elif method == "RRPE":
                alpha, info = rrpe_solve(R, D, starts=[affine_quadratic_coeffs(R, ridge=1e-10)],
                                         tol=2e-6, maxit=800, name="large")
                solve_kkt.append(info["kkt"])
                z = lincomb(X, alpha)
                pred = norm(lincomb(R, alpha)) + sum(D[j] * abs(alpha[j]) for j in range(len(alpha)))
                if pred <= eta * max(0.0, norm(r) - D[-1]):
                    cert += 1
                    x = z
                    took = True
                else:
                    explicit += 1
                    dedicated += 1
                    gz, inner_res_z, inner_it_z = bratu_G_inexact(z, lam, reltol)
                    total_ops += 1
                    inner_total += inner_it_z
                    rz = sub(gz, z)
                    if norm(rz) <= eta * norm(r):
                        explicit_acc += 1
                        x = z
                        took = True
                    else:
                        fallback += 1
                        x = gx
                        took = True
            else:
                raise ValueError(method)
            max_alpha = max(max_alpha, sum(abs(a) for a in alpha))
            if method == "Ridge-RRE":
                explicit += 1
                dedicated += 1
                gz, inner_res_z, inner_it_z = bratu_G_inexact(z, lam, reltol)
                total_ops += 1
                inner_total += inner_it_z
                rz = sub(gz, z)
                if norm(rz) <= eta * norm(r):
                    explicit_acc += 1
                    x = z
                    took = True
                else:
                    fallback += 1
                    x = gx
                    took = True
        if not took:
            x = gx
    return {
        "method": method,
        "n": n,
        "iters": it,
        "ops": total_ops,
        "inner": inner_total,
        "dedicated": dedicated,
        "attempts": attempts,
        "cert": cert,
        "explicit": explicit,
        "explicit_acc": explicit_acc,
        "fallback": fallback,
        "max_alpha": max_alpha,
        "final": final_true,
        "time": time.perf_counter() - t0,
        "max_kkt": max(solve_kkt) if solve_kkt else 0.0,
    }


def large_inexact_table():
    return [run_large_inexact(m) for m in ["Picard", "Ridge-RRE", "RRPE"]]


def noise_floor_once(eps, seed=1):
    rng = random.Random(seed)
    n = 10
    lambdas = [0.985 - 0.015 * i for i in range(n)]
    x = [1.0] * n
    X = []
    Robs = []
    p = 5
    betas = []
    wamps = []
    residuals = []
    prev = None
    for _ in range(45):
        rex = [(lambdas[i] - 1.0) * x[i] for i in range(n)]
        residuals.append(norm(rex))
        noise = rand_vec(n, eps, rng)
        obs = add(rex, noise)
        X.append(x)
        Robs.append(obs)
        if len(X) > p:
            X = X[-p:]
            Robs = Robs[-p:]
        if len(X) >= p:
            alpha, _ = rrpe_solve(Robs, [eps] * p, starts=[prev, affine_quadratic_coeffs(Robs, 1e-10)],
                                  tol=2e-6, maxit=600, name="noise")
            prev = alpha
            z = lincomb(X, alpha)
            pred = norm(lincomb(Robs, alpha)) + eps * sum(abs(a) for a in alpha)
            cur = norm(obs) + eps
            if pred <= 0.98 * cur:
                x = z
                beta = sum(abs(a) for a in alpha)
                betas.append(beta)
                wamps.append(eps * beta)
                continue
        x = [lambdas[i] * x[i] for i in range(n)]
        x = add(x, rand_vec(n, eps, rng))
    tail = residuals[-20:]
    return sum(tail) / len(tail), max(tail), sum(betas) / len(betas), sum(wamps) / len(wamps)


def rand_vec(n, eps, rng):
    if eps == 0:
        return [0.0] * n
    v = [rng.gauss(0.0, 1.0) for _ in range(n)]
    nv = norm(v) or 1.0
    return [eps * x / nv for x in v]


def noise_floor_table(seeds=20):
    rows = []
    for eps in [1e-10, 1e-8, 1e-6, 1e-5, 1e-4]:
        vals = [noise_floor_once(eps, seed=1000 + s) for s in range(seeds)]
        means = [sum(v[i] for v in vals) / seeds for i in range(4)]
        tail_mean_vals = [v[0] for v in vals]
        mean_tail = means[0]
        sd_tail = math.sqrt(sum((x - mean_tail) ** 2 for x in tail_mean_vals) / (seeds - 1))
        max_tail = max(v[1] for v in vals)
        rows.append((eps, mean_tail, sd_tail, max_tail, means[2], means[3]))
    return rows


def ablation_table():
    # Compact close-spectrum ablation with columnwise radii.
    p = 6
    cols = rows_to_cols(close_rows(p, 0.2))
    scalar = [3e-4] * p
    columnwise = [1e-4, 1e-4, 2e-4, 3e-4, 8e-4, 1e-3]
    variants = []
    rre = affine_quadratic_coeffs(cols, 0.0)
    variants.append(("RRE", sum(abs(a) for a in rre), norm(lincomb(cols, rre)), objective(cols, scalar, rre)))
    a_scalar, _ = rrpe_solve(cols, scalar, starts=[rre], tol=3e-7, maxit=8000, name="ablation")
    variants.append(("RRPE scalar", sum(abs(a) for a in a_scalar), norm(lincomb(cols, a_scalar)), objective(cols, scalar, a_scalar)))
    a_col, _ = rrpe_solve(cols, columnwise, starts=[rre, a_scalar], tol=3e-7, maxit=8000, name="ablation")
    variants.append(("RRPE columnwise", sum(abs(a) for a in a_col), norm(lincomb(cols, a_col)), objective(cols, columnwise, a_col)))
    simplex = simplex_coeffs(cols)
    variants.append(("No certification/simplex", sum(abs(a) for a in simplex), norm(lincomb(cols, simplex)), objective(cols, scalar, simplex)))
    ridge = affine_quadratic_coeffs(cols, 1e-6)
    variants.append(("Empirical ridge", sum(abs(a) for a in ridge), norm(lincomb(cols, ridge)), objective(cols, scalar, ridge)))
    return variants


def solver_summary():
    if not SOLVE_LOG:
        return (0, 0, 0, 0, 0)
    return (
        len(SOLVE_LOG),
        max(x["feas"] for x in SOLVE_LOG),
        max(x["kkt"] for x in SOLVE_LOG),
        max(x["objective"] for x in SOLVE_LOG),
        max(x["iters"] for x in SOLVE_LOG),
    )



# The active-set solver below supersedes the first-order primal-dual routine
# above.  Memory sizes in the experiments are small, so exhaustive support/sign
# enumeration is affordable and gives much tighter optimality certificates.

def least_squares_normal(A, b):
    At = list(map(list, zip(*A)))
    AtA = [[sum(At[i][k] * A[k][j] for k in range(len(A))) for j in range(len(At))] for i in range(len(At))]
    Atb = [sum(At[i][k] * b[k] for k in range(len(A))) for i in range(len(At))]
    for ridge in [0.0, 1e-14, 1e-12, 1e-10, 1e-8]:
        B = [row[:] for row in AtA]
        for i in range(len(B)):
            B[i][i] += ridge
        try:
            return gaussian_elimination(B, Atb)
        except Exception:
            pass
    raise ValueError("least-squares solve failed")


def subcolumns(cols, support):
    return [cols[j] for j in support]


def sign_consistent(values, signs, tol=1e-9):
    return all(signs[i] * values[i] >= -tol for i in range(len(values)))


def min_norm_solution(A, b):
    # Return minimum-norm x satisfying A x ~= b for a short-wide A by
    # x=A^T(AA^T)^{-1}b.
    m = len(A)
    if m == 0:
        return []
    AAt = [[sum(A[i][k] * A[j][k] for k in range(len(A[0]))) for j in range(m)] for i in range(m)]
    for ridge in [0.0, 1e-14, 1e-12, 1e-10, 1e-8]:
        B = [row[:] for row in AAt]
        for i in range(m):
            B[i][i] += ridge
        try:
            z = gaussian_elimination(B, b)
            return [sum(A[i][j] * z[i] for i in range(m)) for j in range(len(A[0]))]
        except Exception:
            pass
    raise ValueError("minimum-norm solve failed")


def kkt_residual(cols, delta, alpha):
    p = len(alpha)
    residual = lincomb(cols, alpha)
    nr = norm(residual)
    feas = abs(sum(alpha) - 1.0)
    if nr > 1e-9:
        y = [v / nr for v in residual]
        w = rt_y(cols, y)
        _, stat = best_stationarity_lambda(w, alpha, delta)
        return max(feas, stat)

    active = [j for j, a in enumerate(alpha) if abs(a) > 1e-8]
    if not active:
        return max(feas, 1.0)
    # For zero residual, y may be any vector in the unit ball. Enforce the
    # stationarity equalities on the nonzero coefficients and check interval
    # conditions on zero coefficients.
    A = []
    b = []
    n = len(cols[0])
    for j in active:
        A.append(cols[j][:] + [1.0])
        b.append(-delta[j] * (1.0 if alpha[j] > 0 else -1.0))
    try:
        sol = min_norm_solution(A, b)
    except Exception:
        return max(feas, 1.0)
    y = sol[:n]
    lam = sol[n]
    eq = 0.0
    for row, rhs in zip(A, b):
        eq = max(eq, abs(dot(row, sol) - rhs))
    ball = max(0.0, norm(y) - 1.0)
    interval = 0.0
    for j in range(p):
        val = dot(cols[j], y) + lam
        if abs(alpha[j]) <= 1e-8:
            interval = max(interval, max(0.0, abs(val) - delta[j]))
    return max(feas, eq, ball, interval)


def active_set_candidates(cols, delta):
    p = len(delta)
    n = len(cols[0])
    candidates = []
    for j in range(p):
        a = [0.0] * p
        a[j] = 1.0
        candidates.append(a)
    for signs in itertools.product([-1, 0, 1], repeat=p):
        support = [i for i, sgn in enumerate(signs) if sgn]
        if not support:
            continue
        sub = subcolumns(cols, support)
        local_signs = [signs[i] for i in support]
        k = len(support)

        # Candidate with zero residual on this support, if feasible.
        A = []
        b = []
        for row in range(n):
            A.append([sub[j][row] for j in range(k)])
            b.append(0.0)
        A.append([1.0] * k)
        b.append(1.0)
        try:
            x = least_squares_normal(A, b)
            eq_res = norm([sum(A[i][j] * x[j] for j in range(k)) - b[i] for i in range(len(A))])
            if eq_res <= 1e-8 and sign_consistent(x, local_signs):
                a = [0.0] * p
                for idx, val in zip(support, x):
                    a[idx] = val
                candidates.append(a)
        except Exception:
            pass

        # Smooth nonzero-residual KKT candidate inside this orthant.
        G = gram(sub)
        c = [delta[support[i]] * local_signs[i] for i in range(k)]
        e = [1.0] * k
        try:
            inv_cols = []
            for col in range(k):
                unit = [0.0] * k
                unit[col] = 1.0
                inv_cols.append(gaussian_elimination(G, unit))
            Ginv = [[inv_cols[col][row] for col in range(k)] for row in range(k)]
            Be = [sum(Ginv[i][j] * e[j] for j in range(k)) for i in range(k)]
            Bc = [sum(Ginv[i][j] * c[j] for j in range(k)) for i in range(k)]
            qa = dot(e, Be)
            qb = dot(c, Be)
            qc = dot(c, Bc) - 1.0
            disc = qb * qb - qa * qc
            if disc >= -1e-12 and abs(qa) > 1e-14:
                disc = max(0.0, disc)
                for lam in [(-qb + math.sqrt(disc)) / qa, (-qb - math.sqrt(disc)) / qa]:
                    h = [Bc[i] + lam * Be[i] for i in range(k)]
                    denom = sum(h)
                    if abs(denom) < 1e-14:
                        continue
                    scale = -1.0 / denom
                    if scale <= 0:
                        continue
                    x = [-scale * hi for hi in h]
                    if sign_consistent(x, local_signs):
                        a = [0.0] * p
                        for idx, val in zip(support, x):
                            a[idx] = val
                        candidates.append(a)
        except Exception:
            pass
    return candidates


def rrpe_solve(cols, delta, starts=None, tol=1e-10, maxit=0, name="rrpe"):
    starts = starts or []
    p = len(delta)
    candidates = active_set_candidates(cols, delta)
    for st in starts:
        if st is None or len(st) != p:
            continue
        if abs(sum(st) - 1.0) <= 1e-7:
            candidates.append(st[:])
        else:
            candidates.append(project_affine(st[:]))
    # Remove near-duplicates to keep diagnostics stable.
    unique = []
    seen = set()
    for a in candidates:
        key = tuple(round(x, 10) for x in a)
        if key not in seen and abs(sum(a) - 1.0) <= 1e-6:
            seen.add(key)
            unique.append(a)
    candidates = unique
    best = min(candidates, key=lambda a: objective(cols, delta, a))
    base_best = min([objective(cols, delta, a) for a in candidates] or [objective(cols, delta, best)])
    opt_gap = objective(cols, delta, best) - base_best
    info = {
        "name": name,
        "start": 0,
        "iters": len(candidates),
        "feas": abs(sum(best) - 1.0),
        "objective": objective(cols, delta, best),
        "kkt": kkt_residual(cols, delta, best),
        "dual_norm": 0.0,
        "lambda": 0.0,
        "sanity_gap": opt_gap,
    }
    SOLVE_LOG.append(info)
    return best, info

def fmt(x):
    if isinstance(x, str):
        return x
    if x == 0:
        return "0"
    ax = abs(x)
    if ax < 1e-3 or ax >= 1e4:
        return f"{x:.3e}"
    return f"{x:.4f}"


def print_table(name, rows):
    print("\n" + name)
    for row in rows:
        print(" ".join(fmt(x) for x in row))


def main():
    gauss, pinelis = coverage_tables()
    print_table("CoverageGaussian", gauss)
    print_table("CoveragePinelis", pinelis)
    print_table("Phase", phase_table())
    scaling, slope = scaling_table()
    print_table("Scaling", scaling)
    print("ScalingSlope", fmt(slope))
    bratu = bratu_benchmark()
    print("\nBratuBenchmark")
    for r in bratu:
        print(r["method"], r["iters"], r["ops"], r["dedicated"], r["attempts"], r["cert"],
              r["explicit"], r["explicit_acc"], r["fallback"], fmt(r["max_alpha"]),
              fmt(r["maxC"]), fmt(r["final"]), fmt(r["time"]), fmt(r["max_kkt"]))
    large = large_inexact_table()
    print("\nLargeInexact")
    for r in large:
        print(r["method"], r["n"], r["iters"], r["ops"], r["inner"], r["dedicated"],
              r["attempts"], r["cert"], r["explicit"], r["explicit_acc"], r["fallback"],
              fmt(r["max_alpha"]), fmt(r["final"]), fmt(r["time"]), fmt(r["max_kkt"]))
    print_table("Noise", noise_floor_table(seeds=20))
    print_table("Ablation", ablation_table())
    print("\nSolverSummary", " ".join(fmt(x) for x in solver_summary()))


if __name__ == "__main__":
    main()
