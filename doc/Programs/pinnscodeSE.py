import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ============================================================
# General settings
# ============================================================

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Neural network wavefunction: single hidden layer, tanh
# psi(x) = exp(u_out), u_out is NN output
# ============================================================

class NeuralWaveFunction1D(nn.Module):
    """
    Neural network ansatz for a bosonic wave function in 1D.

    Input:  positions X of shape [batch, N]
    Output: log_psi(X)  (so psi = exp(log_psi))
    """
    def __init__(self, n_particles, n_hidden):
        super().__init__()
        self.n_particles = n_particles
        self.n_hidden = n_hidden

        self.fc1 = nn.Linear(n_particles, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 1)

        # Xavier/Glorot initialization, as in Ref. 27 of the paper
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        """
        x: [batch, N]
        returns log_psi(x): [batch]
        """
        h = torch.tanh(self.fc1(x))
        u_out = self.fc2(h).squeeze(-1)
        # psi = exp(u_out) > 0, but we return log psi for stability
        return u_out  # = log_psi(x)


# ============================================================
# Calogero–Sutherland model in harmonic trap
#   H = sum_i [ -1/2 d^2/dx_i^2 + 1/2 x_i^2 ] +
#       sum_{j<k} beta(beta-1) / (x_j - x_k)^2
# E_exact and psi_exact are known analytically.
# ============================================================

def exact_energy_calogero(N, beta):
    # Eq. (12) in the article
    return N / 2.0 + beta * N * (N - 1) / 2.0

def exact_log_psi_calogero(X, beta, eps=1e-8):
    """
    Exact log wave function, Eq. (11):
    psi_exact = exp(-1/2 sum_i x_i^2) * prod_{j<k} |x_j - x_k|^beta
    X: [batch, N]
    returns log_psi_exact: [batch]
    """
    harmonic = -0.5 * (X ** 2).sum(dim=1)
    # pairwise distances
    batch_size, N = X.shape
    pair_term = X.unsqueeze(2) - X.unsqueeze(1)  # [batch, N, N]
    # take j<k
    idx_j, idx_k = torch.triu_indices(N, N, offset=1)
    deltas = pair_term[:, idx_j, idx_k].abs() + eps
    log_prod = (beta * torch.log(deltas)).sum(dim=1)
    return harmonic + log_prod

def potential_energy_calogero(X, beta, ramp=None, eps=1e-8):
    """
    One-body + interaction potential for Calogero–Sutherland model (Eq. 10).
    X: [batch, N]
    beta: interaction parameter
    ramp: if not None, use the "min[(a n)^2, V_int]" prescription in Eq. (13)
          ramp is the scalar (a n) (already multiplied).
    """
    # harmonic potential: 1/2 sum_i x_i^2
    V_harm = 0.5 * (X ** 2).sum(dim=1)

    batch_size, N = X.shape
    idx_j, idx_k = torch.triu_indices(N, N, offset=1)
    deltas = X[:, idx_j] - X[:, idx_k]  # [batch, n_pairs]
    r2 = deltas ** 2 + eps
    V_int = beta * (beta - 1.0) / r2  # [batch, n_pairs]
    V_int_sum = V_int.sum(dim=1)

    if ramp is not None:
        # Eq. (13): H_int = min[(a n)^2, sum_{j<k} beta(beta-1)/(xj-xk)^2]
        max_val = ramp ** 2
        V_int_sum = torch.minimum(V_int_sum, torch.tensor(max_val, device=X.device))

    return V_harm + V_int_sum


# ============================================================
# Kinetic energy via Laplacian of log psi
#   For psi = exp(u), with u = log psi:
#   (1/psi) d^2 psi/dx_i^2 = (du/dx_i)^2 + d^2u/dx_i^2
#   T = -1/2 sum_i [(du/dx_i)^2 + d^2u/dx_i^2]
# ============================================================

def local_energy_calogero(model, X, beta, ramp=None):
    """
    Compute local energy E_loc(X) = (H psi)(X) / psi(X) for Calogero–Sutherland.

    X: [batch, N] requires_grad=True
    returns E_loc: [batch]
    """
    X = X.clone().detach().requires_grad_(True)
    log_psi = model(X)  # u(X)
    # Gradient du/dx_i
    grad_u = torch.autograd.grad(log_psi.sum(), X, create_graph=True)[0]  # [batch, N]

    laplacian_u = torch.zeros_like(grad_u)
    # Compute d^2u/dx_i^2 by differentiating grad_u[:, i]
    for i in range(X.shape[1]):
        grad_u_i = grad_u[:, i].sum()
        grad2_u_i = torch.autograd.grad(grad_u_i, X, retain_graph=True)[0][:, i]
        laplacian_u[:, i] = grad2_u_i

    # kinetic term: -1/2 sum_i[(du/dx_i)^2 + d^2u/dx_i^2]
    kinetic = -0.5 * ((grad_u ** 2 + laplacian_u).sum(dim=1))

    # potential term
    potential = potential_energy_calogero(X, beta, ramp=ramp)

    E_loc = kinetic + potential
    return E_loc.detach(), log_psi  # detach E_loc; log_psi keeps graph for REINFORCE-like gradient


# ============================================================
# Metropolis sampling with |psi|^2 as target density
# ============================================================

@torch.no_grad()
def metropolis_step(model, X, step_size):
    """
    One Metropolis-Hastings update step.
    X: [n_walkers, N]
    step_size: proposal std-dev
    """
    n_walkers, N = X.shape
    X_prop = X + step_size * torch.randn_like(X)

    log_psi_old = model(X)
    log_psi_new = model(X_prop)

    # acceptance probability: |psi_new|^2 / |psi_old|^2 = exp(2 (log psi_new - log psi_old))
    logA = 2.0 * (log_psi_new - log_psi_old)
    probs = torch.exp(torch.minimum(logA, torch.zeros_like(logA)))  # logA>0 -> prob=1

    accept = (torch.rand_like(probs) < probs).double().unsqueeze(1)
    X_new = accept * X_prop + (1.0 - accept) * X

    return X_new


def metropolis_sample(model, X_init, n_steps, step_size, n_thermal=100, n_skip=10):
    """
    Generate correlated samples using Metropolis-Hastings.

    Returns:
        samples: [n_samples, N]
    """
    X = X_init.clone()
    # Thermalization
    for _ in range(n_thermal):
        X = metropolis_step(model, X, step_size)

    samples = []
    for _ in range(n_steps):
        for __ in range(n_skip):
            X = metropolis_step(model, X, step_size)
        samples.append(X.clone())

    return torch.cat(samples, dim=0)


# ============================================================
# Pretraining to non-interacting ground state (Eq. (8))
# psi_train(x) = exp(-1/2 sum_i x_i^2)
# ============================================================

def pretrain_noninteracting(model,
                            n_particles,
                            n_steps=2000,
                            batch_size=512,
                            lr=1e-3,
                            print_every=200):
    """
    Supervised pretraining: make the network approximate
    psi_train(X) = exp(-1/2 sum_i x_i^2).
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    for step in range(1, n_steps + 1):
        # Sample from simple Gaussian around origin
        X = torch.randn(batch_size, n_particles, device=device)

        # target log psi
        log_psi_target = -0.5 * (X ** 2).sum(dim=1)

        log_psi_model = model(X)
        loss = ((log_psi_model - log_psi_target) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % print_every == 0:
            with torch.no_grad():
                mse = loss.item()
            print(f"[Pretrain] step {step}/{n_steps}, MSE(log psi) = {mse:.4e}")


# ============================================================
# Variational Monte Carlo optimization
# Gradient uses REINFORCE-style estimator:
#   dE/dθ = 2 Cov(E_loc, ∂ log ψ / ∂θ)
# Implemented via loss = mean( (E_loc - E_mean) * log_psi )
# ============================================================

def train_calogero(model,
                   N,
                   beta,
                   n_steps=40000,
                   walkers=256,
                   step_size=0.5,
                   lr=3e-4,
                   ramp_a=1e-3,
                   print_every=500,
                   measure_every=1000):
    """
    Variational Monte Carlo optimization of the Calogero–Sutherland model.

    Returns:
        history: dict with "step", "energy"
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # initial walkers from non-interacting Gaussian
    X_walkers = torch.randn(walkers, N, device=device)

    history = {"step": [], "energy": []}

    E_exact = exact_energy_calogero(N, beta)
    print(f"Exact ground-state energy (analytic): {E_exact:.8f}")

    for step in range(1, n_steps + 1):
        # Interaction ramp parameter (Eq. 13 style)
        ramp = ramp_a * step

        # produce samples for this step
        samples = metropolis_sample(model, X_walkers, n_steps=1,
                                    step_size=step_size, n_thermal=0, n_skip=10)
        X_walkers = samples[-walkers:].clone()  # keep last configuration as new walkers

        # compute local energies and log psi
        E_loc, log_psi = local_energy_calogero(model, samples, beta, ramp=ramp)

        E_mean = E_loc.mean().item()

        # REINFORCE-like gradient: loss = <(E_loc - <E_loc>) log psi>
        E_loc_det = E_loc.detach()
        baseline = E_loc_det.mean()
        loss = ((E_loc_det - baseline) * log_psi).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % print_every == 0:
            print(f"[Train] step {step}/{n_steps}, "
                  f"E_mean = {E_mean:.6f}, "
                  f"(E - E_exact)/E_exact = {(E_mean - E_exact)/E_exact:.3e}")

        if step % measure_every == 0:
            history["step"].append(step)
            history["energy"].append(E_mean)

    return history, X_walkers


# ============================================================
# Fidelity estimate between neural net and exact wavefunction
# using Eq. (15) from the paper:
#   F = <B>^2 / <B^2>, with B = psi_exact / psi_net
# Sampling distribution P(X) ∝ |psi_net|^2
# ============================================================

@torch.no_grad()
def estimate_fidelity(model, N, beta, X_samples, n_eval=10000):
    """
    Estimate fidelity between neural wavefunction and exact Calogero ground state.
    X_samples: [M, N] samples from |psi_net|^2 (Metropolis).
    """
    model.eval()
    if X_samples.shape[0] > n_eval:
        idx = torch.randperm(X_samples.shape[0])[:n_eval]
        X = X_samples[idx].to(device)
    else:
        X = X_samples.to(device)

    log_psi_net = model(X)
    log_psi_exact = exact_log_psi_calogero(X, beta)

    B_log = log_psi_exact - log_psi_net
    B = torch.exp(B_log)

    B_mean = B.mean()
    B2_mean = (B ** 2).mean()
    F = (B_mean ** 2 / B2_mean).item()
    return F


# ============================================================
# Main: run an example close to Fig. 3(a,b,c)
# ============================================================

if __name__ == "__main__":
    # ----------------------------
    # Parameters (you can change)
    # ----------------------------
    N = 5          # number of bosons (e.g. 5 or 8)
    beta = 2.0     # interaction parameter (same as in Fig. 3)
    Nhid = 20      # number of hidden units
    pretrain_steps = 5000
    vmc_steps = 40000
    walkers = 256
    step_size = 0.5
    lr = 3e-4

    print(f"Device: {device}")
    print(f"N = {N}, beta = {beta}, Nhid = {Nhid}")

    # Create model
    model = NeuralWaveFunction1D(N, Nhid).to(device)

    # 1) Pretrain on non-interacting harmonic ground state
    print("\n=== Pretraining to non-interacting ground state ===")
    pretrain_noninteracting(model, n_particles=N,
                            n_steps=pretrain_steps,
                            batch_size=512,
                            lr=1e-3,
                            print_every=500)

    # 2) Variational Monte Carlo optimization with interaction
    print("\n=== Variational Monte Carlo optimization (Calogero–Sutherland) ===")
    history, X_walkers = train_calogero(
        model,
        N=N,
        beta=beta,
        n_steps=vmc_steps,
        walkers=walkers,
        step_size=step_size,
        lr=lr,
        ramp_a=1e-3,
        print_every=1000,
        measure_every=500
    )

    # 3) Plot energy convergence
    steps = np.array(history["step"])
    energies = np.array(history["energy"])
    plt.figure()
    plt.plot(steps, (energies - exact_energy_calogero(N, beta)) /
             exact_energy_calogero(N, beta), marker="o")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Update step")
    plt.ylabel(r"$(E - E_{\rm exact})/E_{\rm exact}$")
    plt.title(f"Energy convergence, N={N}, beta={beta}, Nhid={Nhid}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 4) Estimate fidelity w.r.t. exact wave function
    print("\n=== Estimating fidelity with exact ground state ===")
    # Generate a large set of samples from the trained network
    with torch.no_grad():
        X_init = torch.randn(walkers, N, device=device)
        X_samples = metropolis_sample(model, X_init,
                                      n_steps=200,
                                      step_size=step_size,
                                      n_thermal=100,
                                      n_skip=5)

    F = estimate_fidelity(model, N, beta, X_samples, n_eval=20000)
    print(f"Estimated fidelity F ≈ {F:.4f}")
