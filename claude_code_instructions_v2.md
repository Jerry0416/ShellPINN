# MLS-PINN Implementation Instructions

## Background

This codebase implements Physics-Informed Neural Networks for shell structures, based on Bastek & Kochmann (2023). It uses Naghdi shell theory with a weak form (energy minimization) loss. I need to add stress networks with a normalized connection loss to improve convergence for thin shells.

## Step 1: Analyze the Codebase (DO THIS FIRST, REPORT BEFORE CHANGING ANYTHING)

Read through the entire codebase and report:

1. What is the main entry point / training script?
2. How is the displacement network defined? (class name, architecture, input shape, output shape)
3. How are the 5 output quantities (û₁, û₂, û₃, θ₁, θ₂) produced? (single network with 5 outputs, or separate networks?)
4. How is the trial function φ(ξ) applied to enforce Dirichlet BCs?
5. Where and how are the three strain measures computed?
   - Membrane strain e_αβ
   - Bending strain k_αβ  
   - Shear strain γ_α
6. Where and how are the constitutive stresses computed? Specifically, find the lines that compute:
   - n = t * C : e  (membrane force)
   - m = (t³/12) * C : k  (bending moment)
   - q = κt * D · γ  (shear force)
   Show me the exact code lines.
7. How is the weak form loss (energy functional) assembled? Show the exact expression.
8. What is the thickness parameter t, and how is it used in the code? (variable name, where it appears)
9. How is the optimizer configured? (L-BFGS closure pattern, learning rate, etc.)
10. How are geometric quantities (metric tensor, curvature tensor, Christoffel symbols, √a) stored and accessed?
11. What is the collocation point data structure? (shape, device, requires_grad?)

Do NOT make any changes until I review your analysis and confirm.

## Step 2: Add Stress Networks

After I confirm Step 1, create stress networks following these requirements:

**Architecture**: 8 independent small networks, each outputting a single scalar:
- 3 for normalized membrane force components (ñ₁₁, ñ₁₂, ñ₂₂)
- 3 for normalized bending moment components (m̃₁₁, m̃₁₂, m̃₂₂)  
- 2 for normalized shear force components (q̃₁, q̃₂)

**Each small network**: Same activation function as the existing displacement network, but smaller — 2 hidden layers with 30 neurons each.

**Input**: Same collocation points as the displacement network.

**Output convention**: The networks output NORMALIZED quantities (~O(1) magnitude). Physical stresses are recovered by:
- n_αβ = t × ñ_αβ
- m_αβ = (t³/12) × m̃_αβ
- q_α = κt × q̃_α

**Implementation**: Follow the existing code's patterns and conventions (same device, same dtype, same initialization scheme if any).

## Step 3: Add Normalized Connection Loss

Create a function that computes the normalized connection loss. The mathematical definition is:

L_cnc = (1/N) Σᵢ [ ||ñᵒ - C:e||² + ||m̃ᵒ - C:k||² + ||q̃ᵒ - κD·γ||² ]

Where:
- ñᵒ, m̃ᵒ, q̃ᵒ are the stress network outputs (normalized, ~O(1))
- C:e is the constitutive membrane stress WITHOUT the thickness factor t
- C:k is the constitutive bending stress WITHOUT the thickness factor t³/12
- κD·γ is the constitutive shear stress WITHOUT the thickness factor t

CRITICAL POINT: Look at where the existing code computes the physical stresses (n, m, q). The normalized constitutive stress is the SAME computation but WITHOUT multiplying by the thickness factors. For example, if the code computes `n = t * (some_expression)`, then the normalized constitutive value is just `(some_expression)`.

This loss should have NO thickness factors anywhere. Verify by printing the three sub-terms (membrane, bending, shear) at initialization — they should all be O(1) regardless of the thickness value t.

## Step 4: Add Adaptive Weight Computation

Create a function that computes the adaptive weight w_cnc by:

w_cnc = ||∂L_EF/∂τ_u|| / ||∂L_cnc/∂τ_u||

Where τ_u are the displacement network parameters, L_EF is the existing energy form loss, L_cnc is the normalized connection loss.

Implementation notes:
- Use torch.autograd.grad with retain_graph=True to compute gradients without destroying the computation graph
- Use allow_unused=True in case some parameters don't contribute to one of the losses
- Detach the result — w_cnc should be a plain float, not part of the computation graph
- Add a small epsilon (1e-8) to the denominator to avoid division by zero
- This only needs to be called every ~100 epochs, not every step

## Step 5: Modify the Training Loop

Modify the existing training loop as follows. Keep ALL existing code intact and ADD the new components.

**What to keep unchanged**:
- Displacement network definition and forward pass
- Strain computation
- Energy form loss computation (L_EF / L_weak)
- Trial function application
- Geometric pre-computation
- Collocation point sampling
- FEM reference solution loading (if any)

**What to add**:
1. Instantiate the StressNetworks and move to the same device as the displacement network
2. In each training step, after computing strains from the displacement network:
   a. Forward pass through stress networks to get (ñᵒ, m̃ᵒ, q̃ᵒ)
   b. Compute normalized connection loss L_cnc
   c. Every 100 epochs, update w_cnc using the adaptive weight function
   d. Total loss = L_EF + w_cnc × L_cnc
3. Include stress network parameters in the optimizer

**For the optimizer**: The existing code likely uses L-BFGS with a closure. The closure needs to be modified to include both the energy form loss and the connection loss. The stress network parameters need to be added to the optimizer's parameter list.

**Logging**: Print L_EF, L_cnc, and w_cnc every N epochs so I can monitor training.

## Step 6: Verification

After implementation, add verification code that I can run:

1. **Normalization check**: At epoch 0 (random initialization), print the three sub-terms of L_cnc (membrane, bending, shear parts separately). They should all be O(1) for ANY value of t.

2. **Weight check**: Print w_cnc for t=0.1, t=0.01, t=0.001 (separate runs, just 1 epoch each). w_cnc should be approximately proportional to t.

3. **Baseline recovery**: With w_cnc manually set to 0.0, the results should exactly match the original code (energy form loss only).

## Step 7: Experiment Script

Create a script that runs the comparison experiment:

For the FULLY CLAMPED hyperbolic paraboloid:
- Run baseline (original code, no stress networks) for t/L ∈ {0.1, 0.01, 0.001}
- Run MLS-PINN (with stress networks + normalized connection loss) for the same t/L values
- For each run, record:
  - Loss history (L_EF and L_cnc if applicable) every epoch
  - w_cnc history (if applicable)
  - L2-error vs FEM reference at the end (using the same evaluation method as the original code)
  - Wall clock time per epoch
- Save results to CSV files for later plotting

## Key Principle

At every step, PRESERVE the existing code's behavior. The MLS-PINN modification should be a pure ADDITION — when the stress networks are disabled (w_cnc=0), the code must produce identical results to the original.
