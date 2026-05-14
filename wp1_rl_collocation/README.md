# WP1 — RL-Adaptive Collocation for PINNs
## 2D Poisson PDE · Parametric Source Location Transfer

Pre-train a PPO agent on 20 source locations → zero-shot transfer to 5 unseen positions.

---

## Problem

```
PDE  : −Δu = A · exp(−|(x,y)−(μx,μy)|² / 2σ²)   on [0,1]²
BC   : u = 0 on ∂Ω
PINN : u(x,y) = x(1−x)·y(1−y)·u_NN(x,y)          (hard BC)

A = 10,  σ = 0.05
Training locs : 4×5 grid  (20 source positions)
Test locs     : 5 off-grid positions (zero-shot)
```

---

## Method

| Phase | What happens |
|---|---|
| Pre-train | PPO agent drives collocation on 20 source locations (60 episodes) |
| Freeze | Agent weights saved, no further updates |
| Transfer | Fresh PINN + frozen agent on unseen source location |

**RL state**: 16×16 residual heatmap (flattened, 256-dim)  
**RL action**: 16×16 placement weight map → multinomial sampling  
**Reward**: reduction in L2 error between steps  

---

## Baselines

| Method | Description |
|---|---|
| Uniform | Static random points, never updated |
| RAR | Replace worst-residual points with best candidates |
| PACMANN | Gradient ascent on point positions every step |
| **RL-pretrained** | Frozen PPO policy (this work) |

---

## Results

**Steps to L2 < 0.20** (median, 3 seeds):

| Method | (0.2,0.6) | (0.4,0.4) | (0.6,0.2) | (0.8,0.8) | (0.9,0.4) |
|---|:---:|:---:|:---:|:---:|:---:|
| Uniform | 150 | 100 | 150 | 200 | 200 |
| RAR | 200 | 150 | 200 | 300 | 350 |
| PACMANN | 150 | 150 | 150 | 200 | 250 |
| **RL-pretrained** | **150** | **100** | **150** | **150** | **200** |
| **Speedup vs PACMANN** | 1.00× | **1.50×** | 1.00× | **1.33×** | **1.25×** |

- RAR fails on every geometry (residual spike at step 1 destabilizes training)
- RL outperforms PACMANN on 3/5 unseen geometries — zero re-training

---

## Usage

```bash
pip install -r requirements.txt

# Full run: pre-train on 20 locations, evaluate on 5 unseen (3 seeds, ~30 min)
python geometry_transfer.py

# Skip pre-training (reuse saved checkpoint)
python geometry_transfer.py --skip-pretrain

# Quick smoke test (~2 min)
python geometry_transfer.py --quick

# Custom budget
python geometry_transfer.py --skip-pretrain --n-steps 20 --n-epochs 50 --seeds 3
```

**Output files:**
- `checkpoints/ppo_geometry_agent.pt` — saved PPO weights
- `results/geometry_transfer_results.npz` — convergence curves
- `figures/geometry_transfer_convergence.png` — convergence plot
- `figures/geometry_transfer_weight_maps.png` — RL attention maps

---

## Roadmap

| WP | Description | Status |
|---|---|---|
| **WP1** | Scalar PDE, parametric source location | ✅ This folder |
| **WP2** | Stokes flow, cylinder position transfer | 🔄 `wp2_stokes/` |
| **WP3** | OpenFOAM data + PINN surrogate (same geometry family) | 📋 Planned |
| **WP4** | Geometry generalization to unseen shapes | 🎯 Goal |
