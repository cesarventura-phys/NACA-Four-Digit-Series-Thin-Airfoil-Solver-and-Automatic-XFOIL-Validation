# Thin-Airfoil Theory Solver for NACA 4-Digit Airfoils

**Part II: Software Implementation, Automated XFOIL Coupling, and Validation**

César Carlos Ventura
Departamento de Física, UNS - Av. Alem 1253, (8000) Bahía Blanca, Argentina
📧 cesarventura.phys@gmail.com

---

## Overview

This repository contains a Python implementation of classical thin-airfoil theory for NACA four-digit series airfoils. Given an airfoil code (e.g. `2412`), a chord, a freestream velocity, and an angle of attack, the solver computes the Fourier-series solution of the thin-airfoil integral equation and derives:

- The vortex-sheet strength γ(θ)
- The lift coefficient C_l
- The chordwise loading ΔC_p(x/c)
- Full pressure and velocity fields around the airfoil, plus streamline traces

This is **Part II** of a two-part project. It builds on the [Part I solver](https://github.com/cesarventura-phys/Streamline-and-Pressure-Fields-for-NACA-Four-Digit-Series) by reorganizing it into a modular package and adding:

1. **Polar sweeps** — C_l evaluated as a continuous function of α over a prescribed range.
2. **Automated XFOIL coupling** — the solver drives XFOIL as a scripted subprocess (no manual console interaction) to generate reference polar and surface-pressure data, then automatically overlays it against the thin-airfoil results for validation.

Validation is performed against XFOIL's inviscid panel-method results for both a cambered airfoil (NACA 2412) and a symmetric airfoil (NACA 0012).

---

## Governing Equations

The vortex-sheet strength is represented as the Fourier series

```
γ(θ) = 2 V∞ [ A0 (1+cosθ)/sinθ + Σ_{n=1}^{N} An sin(nθ) ],   θ ∈ [0, π]
```

with the Fourier coefficients obtained analytically from the NACA camber-line slope dy/dx:

```
A0 = α − (1/π) ∫ (dy/dx) dθ
An = (2/π) ∫ (dy/dx) cos(nθ) dθ,   n ≥ 1
```

The lift coefficient follows directly:

```
c_l = π (2A0 + A1),      dc_l/dα = 2π  (per radian)
```

and the chordwise loading (pressure jump across the sheet) is

```
ΔC_p(θ) = C_p,lower − C_p,upper = 2γ(θ) / V∞
```

These three quantities — γ(θ), c_l(α), and ΔC_p(x/c) — are what this project validates against XFOIL.

---

## Repository / Module Structure

| File | Purpose |
|---|---|
| `config.py` | User-defined parameters: airfoil geometry (M, P, C), flow conditions (α, V∞), Fourier truncation order N, visualization grid bounds/resolution, streamline-integration settings. Overridden per call in interactive mode rather than edited in place. |
| `fourier_coefficients.py` | Symbolic derivation of the Fourier coefficients using SymPy: piecewise camber slope defined and integrated analytically over `[0, θp]` and `[θp, π]`, simplified via `cos θp = 1 − 2p`, and compiled to fast numeric functions with `sympy.lambdify` at import time. Symmetric airfoils (m = 0) are special-cased to `A0 = α`, `An = 0`. |
| `vortex.py` | γ(θ), a numerically safe `γ(θ) sinθ` variant that removes the leading-edge singularity, total circulation Γ, and lift coefficient c_l. |
| `velocity.py` | Coordinate transform x(θ)/θ(x), Biot–Savart induced-velocity integrals, total velocity and pointwise C_p, plus a vectorized `compute_field_grid()` for full-mesh evaluation of C_p, u, v. |
| `streamlines.py` | Integrates streamline trajectories dx/dt = u, dy/dt = v. |
| `visualize.py` | Camber-line geometry helper `y_camber()`, filled pressure-contour plots, and a stream-function heatmap reconstructed by direct marching integration of the velocity grid. |
| `xfoil_parser.py` | `read_xfoil_polar()` locates the alpha/CL column header via regex and returns a `pandas.DataFrame`; `read_xfoil_cpwr()` reads a surface-pressure dump and separates upper/lower surfaces by the sign of the local y-coordinate. |
| `xfoil_runner.py` | Parses a NACA four-digit code into (M, P, T) and drives XFOIL as a subprocess to generate both reference datasets, fully unattended. |
| `generate_polar.py` | `compute_polar()` — sweeps α and overlays the thin-airfoil polar with parsed XFOIL data. Parameterized by M, P, and an airfoil label. |
| `generate_loading.py` | `compute_loading()` — evaluates ΔC_p at a single α and overlays it with XFOIL's surface-pressure–derived loading. Parameterized by M, P, and an airfoil label. |
| `main.py` | Original top-level driver for a single airfoil run. |
| `interactive_main.py` | New in Part II — prompts the user for a NACA four-digit code and run parameters, then executes the full solve + XFOIL comparison pipeline. |

---

## Numerical Considerations

- **Leading-edge singularity** (1/sinθ term in γ(θ)): handled analytically by working with the regular product γ(θ)·sinθ, computed by `gamma_theta_safe()` and passed directly to the quadrature integrand — the singularity is never numerically encountered.
- **On-chord induced velocity singularity**: field points falling exactly on y = 0 are perturbed by 10⁻¹⁴ before quadrature.
- **Leading-edge masking**: a small disk of radius `LE_MASK_RADIUS` is masked in all field plots, and `MASK_FRACTION` near the leading/trailing edges is masked in the loading comparison, where both the thin-airfoil singularity and the linear interpolation of discrete XFOIL data become unreliable.

---

## Automated XFOIL Coupling

XFOIL is an interactive, menu-driven console program, but since it reads menu selections from stdin, the exact keystroke sequence a user would type can be written to a buffer and piped directly to its `stdin`. `xfoil_runner.py` builds this buffer for two independent XFOIL subprocess sessions.

### 1. Polar sweep

```
NACA <airfoil code>
GDES

OPERi
ITER 400
PACC
polar_<naca>.txt

ASEQ <alpha_min> <alpha_max> <step>

QUIT
```

Example (NACA 2412, α = −5° to 15° in 1° steps):

```
NACA 2412
GDES

OPERi
ITER 400
PACC
polar_2412.txt

ASEQ -5 15 1

QUIT
```

### 2. Surface-pressure snapshot

```
NACA <airfoil code>
GDES

OPERi
ITER 400
A <alpha>
CPWR cp_<naca>_a<alpha>.txt

QUIT
```

Example (NACA 2412, α = 5°):

```
NACA 2412
GDES

OPERi
ITER 400
A 5
CPWR cp_2412_a5.txt

QUIT
```

In both scripts, the NACA code and α value(s) are interpolated from user input at runtime, so the driver works for any four-digit airfoil and any analysis condition — no manual XFOIL interaction is required.

---

## Validation Results

Validation was performed for the cambered **NACA 2412** (m = 0.02, p = 0.4, t = 0.12) and the symmetric **NACA 0012** (m = 0, p = 0, t = 0.12), both run through `interactive_main.py` with XFOIL reference data generated automatically.

### Lift Curve (Polar) Comparison — NACA 2412

`generate_polar.py` sweeps α from −5° to 15° in 1° increments and overlays the thin-airfoil result with the automatically generated XFOIL polar.

![NACA 2412 polar comparison: thin-airfoil theory vs. XFOIL](results/polar_comparison_2412.png)

Since thin-airfoil theory predicts a lift-curve slope of exactly 2π/rad independent of camber, the thin-airfoil polar is a straight line whose only camber-dependent feature is the zero-lift angle offset. XFOIL's panel-method result shows a similarly linear trend but with a slightly higher slope and higher C_l at every α — expected, since XFOIL retains true airfoil thickness, which increases the effective camber and local leading-edge curvature seen by the flow.

### Chordwise Loading Comparison — NACA 2412 at α = 5°

`generate_loading.py` evaluates ΔC_p(x/c) on a fine θ-grid, masks a small region near the leading/trailing edges, and compares it against ΔC_p obtained from XFOIL's surface-pressure file (upper/lower C_p(x) interpolated separately and subtracted).

![NACA 2412 chordwise loading comparison at alpha = 5 degrees: thin-airfoil theory vs. XFOIL](results/2412_loading_comparison_alpha5.png)

The thin-airfoil loading correctly reproduces the qualitative shape of the XFOIL distribution — a large peak near the leading edge decaying smoothly toward the trailing edge — while systematically underpredicting the magnitude, with the largest discrepancy near the leading edge.

### Summary

| Airfoil | Quantity compared | Outcome |
|---|---|---|
| NACA 2412 | c_l(α), α ∈ [−5°, 15°] | Correct slope trend, small offset |
| NACA 0012 | c_l(α), α ∈ [−5°, 15°] | Correct slope trend, small offset |
| NACA 2412 | ΔC_p(x/c) at α = 5° | Correct shape, underpredicted peak |
| NACA 0012 | ΔC_p(x/c) at α = 5° | Correct shape, underpredicted peak |

The agreement across both airfoils and both validation metrics confirms that the implementation correctly captures the governing inviscid physics. The systematic discrepancies are well understood and arise from the inherent zero-thickness assumption of thin-airfoil theory, not from errors in the numerical implementation.

---

## Conclusions

This project documents the thin-airfoil solver from Part I as working software, automates its comparison against XFOIL end-to-end (no manual reference-data preparation), and extends validation from a single lift-coefficient check to a full lift-curve sweep and a full chordwise loading comparison. Results for both the cambered NACA 2412 and symmetric NACA 0012 airfoils show correct qualitative shape and correct trends with angle of attack and chordwise position, together with a systematic, physically explainable underprediction of magnitude relative to the finite-thickness panel-method solution — confirming the solver meets its design objectives.

---

## References

1. Ventura, C. (2026). *Thin-Airfoil Theory Solver for NACA 4-Digit Airfoils, Part I: Coefficient of Lift, Streamline and Pressure Fields* [Computer software]. GitHub. https://github.com/cesarventura-phys/Streamline-and-Pressure-Fields-for-NACA-Four-Digit-Series
2. Drela, M. (2013). *XFOIL: Subsonic airfoil development system* (Version 6.99) [Computer software]. MIT. https://web.mit.edu/drela/Public/web/xfoil/

---

## Usage

```bash
# Run interactively — prompts for a NACA code and run parameters,
# then executes the full solve + XFOIL comparison pipeline
python interactive_main.py

# Or run the original single-airfoil driver directly
python main.py
```

> **Note:** Running the XFOIL comparison requires a working `xfoil` executable available on your system `PATH`.
