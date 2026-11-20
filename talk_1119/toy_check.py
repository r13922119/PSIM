"""Numerical check of the two-layer toy in claude/dropout_theory.md (not a proof; numpy only).
h = Dropout(W_pre x) + dW x  (inverted dropout on the frozen branch only, coordinate form), margin m = w.h + c, logistic loss.
  A. second- and third-order approximations of the extra expected loss against Monte Carlo;
  B. one layer, squared error: Monte Carlo mean of the stochastic gradient of dW equals the no-dropout gradient;
  C. gradient descent with a fresh mask every step, (i) head only (dW frozen at 0), (ii) head and dW trained together:
     trigger read-out s = -w.b (b orthogonal to every clean frozen output), the clean margin, and their ratio;
     plus the change of s predicted step by step by the second-order expected gradient.
"""
import numpy as np

rng = np.random.default_rng(0)
sig = lambda t: 1 / (1 + np.exp(-t))
softplus = lambda t: np.logaddexp(0, t)
loss = lambda m, y: np.where(y == 1, softplus(-m), softplus(m))

d_in, d_h, n = 12, 30, 400
X = np.zeros((n, d_in)); X[:, :10] = rng.normal(size=(n, 10))
delta = np.zeros(d_in); delta[10] = 1.0
S_u = np.linalg.qr(rng.normal(size=(20, 10)))[0]
b_dir = rng.normal(size=20); b_dir -= S_u @ (S_u.T @ b_dir); b_dir /= np.linalg.norm(b_dir)
W = np.zeros((d_h, d_in)); W[:20, :10] = 1.5 * S_u; W[:20, 10] = 3.0 * b_dir
b = W @ delta
y = (X[:, :10] @ rng.normal(size=10) > 0).astype(int)
U = X @ W.T
print(f"setup: max |u(x).b| over clean inputs = {np.abs(U @ b).max():.1e} (trigger representation orthogonal to clean frozen outputs)")
w0 = 0.3 * rng.normal(size=d_h) - 4 * b / np.linalg.norm(b) ** 2     # head with a backdoor read-out


# ---------------- A
def extra_loss_check(p, w, K=400000):
    lam, e3 = p / (1 - p), -p * (1 - 2 * p) / (1 - p) ** 2             # Var and third moment of e = M/(1-p) - 1
    rows = []
    for i in range(6):
        u, mb, yi = U[i], w @ U[i], y[i]
        M = rng.random((K, d_h)) > p
        eta = ((M / (1 - p) - 1) * u) @ w
        mc = loss(mb + eta, yi).mean() - loss(mb, yi)
        se = loss(mb + eta, yi).std() / np.sqrt(K)
        s1 = sig(mb); l2 = s1 * (1 - s1); l3 = l2 * (1 - 2 * s1) * (1 if yi == 1 else 1)   # d^3/dm^3 of the loss (same for both labels)
        second = 0.5 * l2 * lam * np.sum(w ** 2 * u ** 2)
        third = l3 / 6 * e3 * np.sum(w ** 3 * u ** 3)
        rows.append((mc, se, second, second + third, np.sqrt(lam * np.sum(w ** 2 * u ** 2))))
    return rows


for p, sc in ((0.05, 1.0), (0.2, 1.0), (0.2, 3.0)):
    rows = extra_loss_check(p, sc * w0)
    print(f"A  p={p:<4} head x{sc}: margin-noise sd {min(r[4] for r in rows):.2f}-{max(r[4] for r in rows):.2f}")
    for mc, se, s2, s3, sd in rows[:3]:
        print(f"     Monte Carlo {mc:.5f} (s.e. {se:.5f}) | 2nd order {s2:.5f} | 2nd+3rd order {s3:.5f}")

# ---------------- B
p = 0.2
dW = 0.1 * rng.normal(size=(d_h, d_in)); Y = rng.normal(size=(n, d_h))
x, u, t = X[0], U[0], Y[0]
g0 = -np.outer(t - u - dW @ x, x)
K = 200000
xi = ((rng.random((K, d_h)) > p) / (1 - p) - 1) * u
gm = g0 + np.outer(xi.mean(0), x)                      # mean of -(t - u - dW x - xi) x^T over the masks
se = (xi.std(0)[:, None] * np.abs(x)[None, :]).max() / np.sqrt(K)
print(f"B  squared error: max |mean stochastic grad - no-dropout grad| = {np.abs(gm - g0).max():.1e}; largest Monte Carlo s.e. {se:.1e}")


# ---------------- C
def train(p, train_dW, steps=4000, eta=0.05, seed=1):
    r = np.random.default_rng(seed)
    lam = p / (1 - p)
    w, c, dW = w0.copy(), 0.0, np.zeros((d_h, d_in))
    s_pred = -w @ b
    for it in range(steps):
        Hb = U + X @ dW.T                                   # hidden states without masks
        mb = Hb @ w + c
        q = sig(mb) * (1 - sig(mb)); qp = q * (1 - 2 * sig(mb))
        if p > 0:   # second-order expected gradient of w (data term at the mean, D w, and the q' term)
            var = lam * (U ** 2) @ (w ** 2)
            grad_pred = Hb.T @ (sig(mb) - y) / n + lam * (q[:, None] * U ** 2).mean(0) * w + Hb.T @ (0.5 * qp * var) / n
        else:
            grad_pred = Hb.T @ (sig(mb) - y) / n
        s_pred += eta * (b @ grad_pred)
        M = (r.random((n, d_h)) > p) / (1 - p) if p > 0 else 1.0
        H = M * U + X @ dW.T
        g = sig(H @ w + c) - y
        gw, gc = H.T @ g / n, g.mean()
        if train_dW:
            dW -= eta * np.outer(w, (g[:, None] * X).mean(0))
        w -= eta * gw; c -= eta * gc
    Hb = U + X @ dW.T; m = Hb @ w + c
    pos = y == 1
    s = -w @ b
    return s, s_pred, np.median(m[pos]), (m * (2 * y - 1) > 0).mean(), 0.0, np.abs(dW @ delta).max()


for train_dW, name in ((False, "head only (dW = 0)"), (True, "head and dW together")):
    print(f"C  {name}:")
    for p in (0.0, 0.05, 0.1, 0.2):
        s, sp, med_m, acc, wb, leak = train(p, train_dW)
        print(f"     p={p:<4} pull s {(-w0 @ b):.2f} -> {s:.2f} (2nd-order prediction {sp:.2f}); clean positive median margin {med_m:.2f}; "
              f"s / margin {s / med_m:.3f}; train acc {acc:.3f}; |dW delta| {leak:.0e}")
