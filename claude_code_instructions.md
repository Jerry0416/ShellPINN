# MLS-PINN Implementation Instructions for Claude Code

## Overview

I have a working PyTorch codebase for Physics-Informed Neural Networks (PINNs) applied to shell structures, based on the paper "Physics-Informed Neural Networks for shell structures" by Bastek & Kochmann (2023). The code is from the GitHub repository: https://github.com/jhbastek/PhysicsInformedShellStructures

I need you to modify this codebase to implement a "Mixed-Loss Shell PINN" (MLS-PINN) framework. The modification adds stress networks with normalized connection losses to improve training convergence for thin shells.

## Step 1: Analyze the Existing Codebase

Before making any changes, please:

1. Read through the entire codebase and understand:
   - The main training script and how the weak form loss (energy functional) is computed
   - How the displacement network is defined (architecture, activation, input/output)
   - How strains (e_αβ, k_αβ, γ_α) are computed from displacements using automatic differentiation
   - How geometric quantities (metric tensor a_αβ, curvature b_αβ, Christoffel symbols Γ, √a) are pre-computed from the chart φ
   - How the constitutive tensors C^αβσρ and D^αβ are defined
   - How the trial function φ(ξ) enforces Dirichlet boundary conditions
   - How collocation points are sampled (Sobol sequence)
   - How the L-BFGS optimizer is configured

2. Identify the key functions/classes for:
   - Forward pass of the displacement network
   - Strain computation (membrane strain e, bending strain k, shear strain γ)
   - Constitutive stress computation (n = t*C:e, m = t³/12*C:k, q = κt*D·γ)
   - Energy functional computation (the weak form loss L_weak)
   - Training loop

3. Report back what you found before proceeding to modifications.

## Step 2: Implement Stress Networks

### 2.1 Create Stress Network Module

Create a new module `StressNetworks` that contains 8 independent small networks, one for each normalized stress component:

```python
class SmallMLP(nn.Module):
    """Small MLP for a single stress component output."""
    def __init__(self, input_dim=2, hidden_dim=30, num_layers=2, activation=nn.GELU):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(activation())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation())
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class StressNetworks(nn.Module):
    """
    8 independent small networks outputting NORMALIZED stress components.
    
    Outputs (all ~O(1) magnitude):
        n_tilde_11, n_tilde_12, n_tilde_22  (normalized membrane forces)
        m_tilde_11, m_tilde_12, m_tilde_22  (normalized bending moments)
        q_tilde_1, q_tilde_2                (normalized shear forces)
    
    Physical stresses are recovered via:
        n_αβ = t * n_tilde_αβ
        m_αβ = (t³/12) * m_tilde_αβ
        q_α  = κ*t * q_tilde_α
    """
    def __init__(self, input_dim=2, hidden_dim=30, num_layers=2):
        super().__init__()
        # Membrane force components (normalized)
        self.net_n11 = SmallMLP(input_dim, hidden_dim, num_layers)
        self.net_n12 = SmallMLP(input_dim, hidden_dim, num_layers)
        self.net_n22 = SmallMLP(input_dim, hidden_dim, num_layers)
        # Bending moment components (normalized)
        self.net_m11 = SmallMLP(input_dim, hidden_dim, num_layers)
        self.net_m12 = SmallMLP(input_dim, hidden_dim, num_layers)
        self.net_m22 = SmallMLP(input_dim, hidden_dim, num_layers)
        # Shear force components (normalized)
        self.net_q1 = SmallMLP(input_dim, hidden_dim, num_layers)
        self.net_q2 = SmallMLP(input_dim, hidden_dim, num_layers)
    
    def forward(self, xi):
        """
        Args:
            xi: collocation points (N, 2) in reference domain
        Returns:
            n_tilde: (N, 3) normalized membrane forces [n11, n12, n22]
            m_tilde: (N, 3) normalized bending moments [m11, m12, m22]
            q_tilde: (N, 2) normalized shear forces [q1, q2]
        """
        n_tilde = torch.cat([
            self.net_n11(xi),
            self.net_n12(xi),
            self.net_n22(xi)
        ], dim=-1)
        
        m_tilde = torch.cat([
            self.net_m11(xi),
            self.net_m12(xi),
            self.net_m22(xi)
        ], dim=-1)
        
        q_tilde = torch.cat([
            self.net_q1(xi),
            self.net_q2(xi)
        ], dim=-1)
        
        return n_tilde, m_tilde, q_tilde
```

### 2.2 Placement

Place the `StressNetworks` class in the same file as the existing displacement network, or in a new file `stress_networks.py` depending on the code structure.

## Step 3: Implement Normalized Connection Loss

### 3.1 Connection Loss Function

Create a function that computes the normalized connection loss. This function needs access to:
- The strain fields (e, k, γ) computed from the displacement network
- The normalized stress outputs (n_tilde, m_tilde, q_tilde) from the stress networks
- The constitutive tensors C and D

```python
def compute_normalized_connection_loss(n_tilde, m_tilde, q_tilde,
                                        e_membrane, k_bending, gamma_shear,
                                        C_tensor, D_tensor, kappa=5.0/6.0):
    """
    Normalized connection loss: compares normalized stress network outputs
    with constitutive stresses computed from displacement network.
    
    CRITICAL: No thickness factors appear in this loss.
    
    Args:
        n_tilde: (N, 3) normalized membrane forces from stress network
        m_tilde: (N, 3) normalized bending moments from stress network  
        q_tilde: (N, 2) normalized shear forces from stress network
        e_membrane: (N, 3) membrane strain from displacement network [e11, e12, e22]
        k_bending: (N, 3) bending strain from displacement network [k11, k12, k22]
        gamma_shear: (N, 2) shear strain from displacement network [γ1, γ2]
        C_tensor: constitutive tensor for membrane/bending (function of position)
        D_tensor: constitutive tensor for shear (function of position)
        kappa: shear correction factor (5/6)
    
    Returns:
        L_cnc: scalar, normalized connection loss
    """
    # Constitutive stresses (normalized, NO thickness factors)
    # n_constitutive = C : e  (this would give n/t, i.e., the normalized value)
    # m_constitutive = C : k  (this would give 12*m/t³, i.e., the normalized value)
    # q_constitutive = κ * D · γ  (this would give q/(κt), but we keep κ)
    
    # IMPORTANT: The existing code likely computes C:e as part of the stress calculation.
    # We need ONLY the C:e part, WITHOUT the thickness factor t.
    # Look at how the existing code computes σ = t * C : e and extract just C : e.
    
    n_constitutive = compute_C_double_contract_e(C_tensor, e_membrane)  # = C:e, no t
    m_constitutive = compute_C_double_contract_k(C_tensor, k_bending)   # = C:k, no t³/12
    q_constitutive = kappa * compute_D_dot_gamma(D_tensor, gamma_shear) # = κD·γ, no t
    
    # MSE for each component
    L_cnc_n = torch.mean((n_tilde - n_constitutive)**2)
    L_cnc_m = torch.mean((m_tilde - m_constitutive)**2)
    L_cnc_q = torch.mean((q_tilde - q_constitutive)**2)
    
    return L_cnc_n + L_cnc_m + L_cnc_q
```

### 3.2 CRITICAL Implementation Note

The key insight is that the normalized connection loss compares:
- `n_tilde` (network output, ~O(1)) with `C:e` (constitutive, ~O(1))
- `m_tilde` (network output, ~O(1)) with `C:k` (constitutive, ~O(1))
- `q_tilde` (network output, ~O(1)) with `κD·γ` (constitutive, ~O(1))

The existing code likely computes `n = t * C:e` (physical membrane force). You need to extract `C:e` WITHOUT the thickness factor. Look for where the constitutive relation is applied and factor out `t`.

For example, if existing code has:
```python
sigma = t * C_voigt @ elastic_strain  
```
Then the normalized constitutive stress is simply:
```python
sigma_normalized = C_voigt @ elastic_strain  # without t
```

Similarly for bending moments, if existing code has:
```python
moment = (t**3 / 12) * C_voigt @ bending_strain
```
Then:
```python
moment_normalized = C_voigt @ bending_strain  # without t³/12
```

## Step 4: Implement Adaptive Weight

```python
def compute_adaptive_w_cnc(displacement_model, L_EF, L_cnc, epsilon=1e-8):
    """
    Compute adaptive weight for connection loss based on gradient norm balancing.
    
    w_cnc = ||grad(L_EF)||  /  ||grad(L_cnc)||
    
    This ensures the two gradient pathways contribute equally to parameter updates.
    Theoretically, w_cnc ~ O(t) for thin shells.
    
    Args:
        displacement_model: the displacement network
        L_EF: energy form loss (scalar)
        L_cnc: normalized connection loss (scalar)
        epsilon: small constant to prevent division by zero
    
    Returns:
        w_cnc: adaptive weight (scalar, detached from computation graph)
    """
    # Gradient of energy form w.r.t. displacement network parameters
    grads_EF = torch.autograd.grad(
        L_EF, displacement_model.parameters(),
        retain_graph=True, create_graph=False, allow_unused=True
    )
    norm_EF = torch.sqrt(
        sum(g.detach().norm()**2 for g in grads_EF if g is not None)
    )
    
    # Gradient of connection loss w.r.t. displacement network parameters
    grads_cnc = torch.autograd.grad(
        L_cnc, displacement_model.parameters(),
        retain_graph=True, create_graph=False, allow_unused=True
    )
    norm_cnc = torch.sqrt(
        sum(g.detach().norm()**2 for g in grads_cnc if g is not None)
    )
    
    w_cnc = (norm_EF / (norm_cnc + epsilon)).item()
    return w_cnc
```

## Step 5: Modify Training Loop

### 5.1 Joint Training (simplest version, implement first)

Modify the existing training loop to include the stress networks and connection loss:

```python
# === Initialization ===
# Keep existing displacement network exactly as-is
displacement_net = ...  # existing code

# Add stress networks
stress_net = StressNetworks(input_dim=2, hidden_dim=30, num_layers=2).to(device)

# Separate optimizers
optimizer_u = torch.optim.LBFGS(
    displacement_net.parameters(),
    lr=1.0,
    max_iter=20,
    line_search_fn='strong_wolfe'
)
optimizer_sigma = torch.optim.LBFGS(
    stress_net.parameters(),
    lr=1.0,
    max_iter=20,
    line_search_fn='strong_wolfe'
)

# Adaptive weight (will be updated during training)
w_cnc = 1.0  # initial value
w_cnc_update_freq = 100  # update every 100 epochs

# === Training Loop ===
for epoch in range(num_epochs):
    
    # --- Forward pass: displacement network (existing code) ---
    u_hat, theta = displacement_net(xi_collocation)
    # Apply trial function for Dirichlet BCs (existing code)
    u_hat = phi_trial * u_hat
    theta = phi_trial * theta
    
    # Compute strains (existing code, should already exist)
    e_membrane = compute_membrane_strain(u_hat, theta, xi_collocation, geo_data)
    k_bending = compute_bending_strain(u_hat, theta, xi_collocation, geo_data)
    gamma_shear = compute_shear_strain(u_hat, theta, xi_collocation, geo_data)
    
    # Compute energy form loss (existing code, keep exactly as-is)
    L_EF = compute_weak_form_loss(e_membrane, k_bending, gamma_shear, 
                                   C_tensor, D_tensor, t, kappa, sqrt_a, W_ext)
    
    # --- Forward pass: stress networks (NEW) ---
    n_tilde, m_tilde, q_tilde = stress_net(xi_collocation)
    
    # --- Normalized connection loss (NEW) ---
    L_cnc = compute_normalized_connection_loss(
        n_tilde, m_tilde, q_tilde,
        e_membrane, k_bending, gamma_shear,
        C_tensor, D_tensor, kappa
    )
    
    # --- Adaptive weight update (NEW) ---
    if epoch % w_cnc_update_freq == 0 and epoch > 0:
        w_cnc = compute_adaptive_w_cnc(displacement_net, L_EF, L_cnc)
    
    # --- Total loss ---
    L_total = L_EF + w_cnc * L_cnc
    
    # --- Optimization step ---
    # For L-BFGS, you need closures. The implementation depends on 
    # how the existing code handles L-BFGS.
    
    # Option A: If existing code uses a single optimizer
    # Combine all parameters:
    all_params = list(displacement_net.parameters()) + list(stress_net.parameters())
    optimizer_all = torch.optim.LBFGS(all_params, ...)
    
    def closure():
        optimizer_all.zero_grad()
        # Recompute forward pass and losses inside closure
        u_hat, theta = displacement_net(xi_collocation)
        u_hat = phi_trial * u_hat
        theta = phi_trial * theta
        e = compute_membrane_strain(...)
        k = compute_bending_strain(...)
        gamma = compute_shear_strain(...)
        L_EF = compute_weak_form_loss(...)
        n_t, m_t, q_t = stress_net(xi_collocation)
        L_cnc = compute_normalized_connection_loss(n_t, m_t, q_t, e, k, gamma, ...)
        loss = L_EF + w_cnc * L_cnc
        loss.backward()
        return loss
    
    optimizer_all.step(closure)
    
    # --- Logging ---
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: L_EF={L_EF.item():.6e}, "
              f"L_cnc={L_cnc.item():.6e}, w_cnc={w_cnc:.4f}, "
              f"L_total={L_total.item():.6e}")
```

### 5.2 Important: Preserve Existing Code

Do NOT modify:
- The displacement network architecture
- The strain computation functions
- The energy form loss computation
- The geometric pre-computation
- The collocation point sampling
- The trial function definition

Only ADD:
- StressNetworks class
- Normalized connection loss function
- Adaptive weight function
- Modifications to the training loop to include the new loss

## Step 6: Experiment Configuration

### 6.1 Baseline Experiment (no stress networks)

First, run the existing code as-is for the fully clamped hyperbolic paraboloid with:
- t/L = 0.1, N_c = 2048, 100 epochs
- Save: loss history, L2-errors vs FEM, training time per epoch

### 6.2 MLS-PINN Experiment

Run the modified code with stress networks and normalized connection loss:
- Same geometry, BCs, N_c, and number of epochs
- Save: L_EF history, L_cnc history, w_cnc history, L2-errors vs FEM, training time per epoch

### 6.3 Thickness Sweep

For BOTH baseline and MLS-PINN, run with:
- t/L ∈ {0.1, 0.01, 0.001}
- Record epochs needed to reach average L2-error < 5%

### 6.4 Output Files

Save results as:
```
results/
  baseline_t0.1/    (loss_history.csv, l2_errors.csv)
  baseline_t0.01/
  baseline_t0.001/
  mlspinn_t0.1/     (loss_EF.csv, loss_cnc.csv, w_cnc.csv, l2_errors.csv)
  mlspinn_t0.01/
  mlspinn_t0.001/
```

## Step 7: Verification Checklist

After implementation, verify:

1. **Connection loss is truly normalized**: Print the three sub-components of L_cnc (membrane, bending, shear). They should all be O(1) magnitude at initialization, regardless of thickness t.

2. **Adaptive weight behaves as expected**: Print w_cnc values. For t/L=0.1, w_cnc should be ~0.1. For t/L=0.01, w_cnc should be ~0.01. For t/L=0.001, w_cnc should be ~0.001. If w_cnc ~ O(t), the theory is confirmed.

3. **Baseline is not broken**: With w_cnc=0 (disable connection loss), results should exactly match the original code.

4. **Physical solution is unchanged**: At convergence, verify that:
   - t * n_tilde ≈ n (physical membrane force from displacement network via constitutive relation)
   - (t³/12) * m_tilde ≈ m (physical bending moment from displacement network)
   This confirms the connection loss drives the stress network to the correct physical solution.

## Summary of Files to Create/Modify

1. **NEW FILE** (or add to existing): `stress_networks.py` containing `SmallMLP`, `StressNetworks`
2. **NEW FILE** (or add to existing): `mls_losses.py` containing `compute_normalized_connection_loss`, `compute_adaptive_w_cnc`
3. **MODIFY**: Main training script to incorporate stress networks and new losses
4. **NEW FILE**: `run_experiments.py` for systematic comparison experiments

Please start by analyzing the existing codebase structure and report what you find before making any changes.
