import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

torch.set_default_dtype(torch.float64)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

REGION = 2.0


def f_true(x1):
    return 2.0 + torch.sin(x1)


def f_nom(x1):
    return 0.0 * x1


class RBFNet(torch.nn.Module):
    def __init__(self, centers, width, use_bias=True):
        super().__init__()
        self.register_buffer("centers", centers)
        self.width = width
        self.use_bias = use_bias

    def forward(self, x):
        g = torch.exp(-((x - self.centers) ** 2) / (2.0 * self.width ** 2))
        if self.use_bias:
            return torch.cat([torch.ones_like(x)[None], g])
        return g

    def d_dx(self, x):
        d = self.forward(x) * torch.cat([torch.zeros_like(x)[None],
                                         -(x - self.centers) / self.width ** 2])
        return d


class AdaptiveNeuralController(torch.nn.Module):
    def __init__(self, n_centers=20, width=0.4, centers_lo=-2.0, centers_hi=2.0,
                 eps=0.1, sigma=0.1, gamma=5.0, Gamma=5.0, psi0=0.0,
                 use_bias=True):
        super().__init__()
        centers = torch.linspace(centers_lo, centers_hi, n_centers)
        self.net = RBFNet(centers, width, use_bias)
        self.n = n_centers + (1 if use_bias else 0)
        self.eps = eps
        self.sigma = sigma
        self.gamma = gamma
        self.Gamma = Gamma
        self.psi0 = psi0
        self.theta0 = torch.zeros(self.n)

    @torch.no_grad()
    def control_and_adapt(self, s):
        x1, x2 = s[0], s[1]
        theta = s[2:2 + self.n]
        psi = s[2 + self.n]

        zeta = self.net(x1)
        dzeta = self.net.d_dx(x1)
        w1 = torch.tanh(x1 / self.eps)
        sech2 = 1.0 - w1 ** 2
        f = f_nom(x1)

        alpha = -x1 - f - torch.dot(theta, zeta) - psi * w1
        alpha_x1 = -1.0 - torch.dot(theta, dzeta) - (psi / self.eps) * sech2
        alpha_theta = -zeta
        alpha_psi = -w1

        z1 = x1
        z2 = x2 - alpha

        w2 = alpha_x1 * torch.tanh(alpha_x1 * z2 / self.eps)
        dtheta = self.Gamma * (zeta * z1 - z2 * alpha_x1 * zeta
                               - self.sigma * (theta - self.theta0))
        dpsi = self.gamma * (z1 * w1 + z2 * w2 - self.sigma * (psi - self.psi0))

        beta2 = psi * w2
        u = (-z1 - z2
             + alpha_x1 * (x2 + f + torch.dot(theta, zeta))
             + torch.dot(alpha_theta, dtheta)
             + alpha_psi * dpsi
             - beta2)
        return u, dtheta, dpsi, z1, z2


def make_rhs(ctrl, adapt):
    def rhs(t, s_np):
        s = torch.as_tensor(s_np)
        u, dtheta, dpsi, _, _ = ctrl.control_and_adapt(s)
        if not adapt:
            dtheta = torch.zeros_like(dtheta)
            dpsi = torch.zeros_like(dpsi)
        d = torch.empty_like(s)
        d[0] = s[1] + f_true(s[0])
        d[1] = u
        d[2:2 + ctrl.n] = dtheta
        d[2 + ctrl.n] = dpsi
        return d.numpy()
    return rhs


def simulate(T=20.0, adapt=True, x0=(1.5, 0.0), n_grid=4000, **ctrl_kwargs):
    ctrl = AdaptiveNeuralController(**ctrl_kwargs)
    n = ctrl.n
    s0 = np.zeros(2 + n + 1)
    s0[0], s0[1] = x0
    sol = solve_ivp(make_rhs(ctrl, adapt), (0.0, T), s0, method="DOP853",
                    rtol=1e-9, atol=1e-11, dense_output=True, max_step=0.05)
    t = np.linspace(0.0, T, n_grid)
    s = sol.sol(t)

    z1 = s[0].copy()
    z2 = np.empty_like(t)
    u = np.empty_like(t)
    with torch.no_grad():
        for i in range(n_grid):
            si = torch.as_tensor(s[:, i])
            ui, _, _, _, z2i = ctrl.control_and_adapt(si)
            u[i] = ui.item()
            z2[i] = z2i.item()
    theta = s[2:2 + n]
    psi = s[2 + n]
    log = {
        "t": t, "x1": s[0], "x2": s[1], "u": u, "z1": z1, "z2": z2,
        "theta": theta, "psi": psi,
        "theta_norm": np.linalg.norm(theta, axis=0),
    }
    return ctrl, log


def ideal_parameters(ctrl, region, npts=4001):
    x = torch.linspace(-region, region, npts)
    Z = torch.stack([ctrl.net(xi) for xi in x])
    phi = f_true(x) - f_nom(x)
    w = 2 * region / (npts - 1)
    theta_star = torch.linalg.solve((Z.T @ Z) * w + 1e-6 * torch.eye(ctrl.n),
                                    (Z.T @ phi) * w)
    delta = phi - Z @ theta_star
    phi_star = delta.abs().max().item()
    return theta_star, phi_star


def lyapunov(ctrl, log, theta_star, psi_star):
    Gamma_inv = 1.0 / ctrl.Gamma
    gamma_inv = 1.0 / ctrl.gamma
    dt_ = log["theta"] - theta_star.numpy()[:, None]
    V = 0.5 * (log["z1"] ** 2 + log["z2"] ** 2
               + Gamma_inv * np.sum(dt_ ** 2, axis=0)
               + gamma_inv * (log["psi"] - psi_star) ** 2)
    return V


def main():
    T = 20.0
    kw = dict(n_centers=9, width=0.8, centers_lo=-REGION, centers_hi=REGION,
              eps=0.1, sigma=0.1, gamma=5.0, Gamma=5.0, use_bias=True)

    ctrl, log = simulate(T=T, adapt=True, x0=(1.5, 0.0), **kw)
    _, log_nom = simulate(T=T, adapt=False, x0=(1.5, 0.0), **kw)

    theta_star, phi_star = ideal_parameters(ctrl, REGION)
    psi_star = max(phi_star, 0.0)
    V = lyapunov(ctrl, log, theta_star, psi_star)

    c = min(2.0, ctrl.sigma * ctrl.Gamma, ctrl.sigma * ctrl.gamma)
    X = (2 * 0.2785 * ctrl.eps * phi_star ** 2
         + 0.5 * ctrl.sigma * ((torch.linalg.norm(theta_star) ** 2).item() + psi_star ** 2))
    p_bound = X / c

    print("=== Model ===")
    print("dx1 = x2 + f*(x1),  dx2 = u,   f*(x1) = 2 + sin(x1),  f(x1) = 0")
    print(f"RBF: n={ctrl.n}, width={kw['width']}, region=[-{REGION},{REGION}]")
    print(f"Design: eps={ctrl.eps}, sigma={ctrl.sigma}, Gamma={ctrl.Gamma}, gamma={ctrl.gamma}")
    print()
    print("=== Reconstruction error (Assumption 1) ===")
    print(f"ideal ||theta*|| = {torch.linalg.norm(theta_star).item():.4f}")
    print(f"phi* = max |delta(x1)| on R = {phi_star:.4e}")
    print()
    tail = log["t"] >= (T - 2.0)
    print("=== Adaptive neural control ===")
    print(f"final x1                       : {log['x1'][-1]: .3e}")
    print(f"max |x1| on [T-2,T]            : {np.max(np.abs(log['x1'][tail])):.4e}")
    print(f"max |x2| on [T-2,T]            : {np.max(np.abs(log['x2'][tail])):.4e}")
    print(f"peak |u|                       : {np.max(np.abs(log['u'])):.3f}")
    print(f"final ||theta||, psi           : {log['theta_norm'][-1]:.4f}, {log['psi'][-1]:.4f}")
    print(f"V(0) -> V(T)                   : {V[0]:.4f} -> {V[-1]:.4e}")
    print(f"theoretical ultimate bound p   : {p_bound:.4f}   (c={c:.3f}, X={X:.4f})")
    print()
    print("=== Non-adaptive (NN off, same control law) ===")
    print(f"max |x1| on [T-2,T]            : {np.max(np.abs(log_nom['x1'][tail])):.4e}")
    print(f"max |x2| on [T-2,T]            : {np.max(np.abs(log_nom['x2'][tail])):.4e}")

    check = boundedness_check(ctrl, kw)

    fig, ax = plt.subplots(3, 2, figsize=(12, 10))
    ax[0, 0].plot(log["t"], log["x1"], label="adaptive NN")
    ax[0, 0].plot(log_nom["t"], log_nom["x1"], "--", label="no adaptation")
    ax[0, 0].axhline(np.max(np.abs(log["x1"][tail])), color="g", ls=":", lw=1)
    ax[0, 0].set_title("Output $x_1=y$")
    ax[0, 0].set_xlabel("t [s]"); ax[0, 0].legend(); ax[0, 0].grid(True)

    ax[0, 1].plot(log["t"], log["x2"])
    ax[0, 1].set_title("State $x_2$")
    ax[0, 1].set_xlabel("t [s]"); ax[0, 1].grid(True)

    ax[1, 0].plot(log["t"], log["u"])
    ax[1, 0].set_title("Control input $u$ (smooth)")
    ax[1, 0].set_xlabel("t [s]"); ax[1, 0].grid(True)

    ax[1, 1].plot(log["t"], log["theta_norm"], label=r"$\|\theta\|$")
    ax[1, 1].plot(log["t"], log["psi"], label=r"$\psi$")
    ax[1, 1].plot(log["t"], log["psi"] * 0 + phi_star, "k:", label=r"$\phi^*$")
    ax[1, 1].set_title("Parameter estimates (bounded)")
    ax[1, 1].set_xlabel("t [s]"); ax[1, 1].legend(); ax[1, 1].grid(True)

    ax[2, 0].plot(log["t"], log["z1"], label="$z_1$")
    ax[2, 0].plot(log["t"], log["z2"], label="$z_2$")
    ax[2, 0].set_title("Error coordinates")
    ax[2, 0].set_xlabel("t [s]"); ax[2, 0].legend(); ax[2, 0].grid(True)

    ax[2, 1].semilogy(log["t"], np.maximum(V, 1e-18))
    ax[2, 1].axhline(p_bound, color="r", ls="--", label=f"$p={p_bound:.2f}$")
    ax[2, 1].set_title("Lyapunov function $V(t)\\leq p+(V(0)-p)e^{-ct}$")
    ax[2, 1].set_xlabel("t [s]"); ax[2, 1].legend(); ax[2, 1].grid(True)
    fig.tight_layout()
    path = os.path.join(OUT, "polycarpou_anc_results.png")
    fig.savefig(path, dpi=130)
    print(f"\nSaved figure: {path}")

    fig2 = plt.figure(figsize=(7, 5))
    xs = np.linspace(-REGION, REGION, 400)
    with torch.no_grad():
        approx = np.array([float(torch.dot(theta_star, ctrl.net(torch.tensor(x)))) for x in xs])
    plt.plot(xs, f_true(torch.tensor(xs)).numpy(), label=r"$f^*(x_1)$")
    plt.plot(xs, approx, "--", label=r"$\theta^{*T}\zeta(x_1)$ (RBF)")
    plt.title(f"Best RBF approximation, $\\phi^*$={phi_star:.2e}")
    plt.xlabel("$x_1$"); plt.legend(); plt.grid(True)
    fig2.tight_layout()
    path2 = os.path.join(OUT, "polycarpou_anc_approx.png")
    fig2.savefig(path2, dpi=130)
    print(f"Saved figure: {path2}")


def boundedness_check(ctrl, kw):
    print()
    print("=== Semiglobal check: different initial conditions ===")
    for x0 in [1.0, 1.5, 2.0]:
        _, l = simulate(T=20.0, adapt=True, x0=(x0, 0.0), **kw)
        t = l["t"] >= 18
        print(f"  x0=({x0},0): max|x1|_tail={np.max(np.abs(l['x1'][t])):.4e}  "
              f"peak|u|={np.max(np.abs(l['u'])):.2f}")


if __name__ == "__main__":
    main()
