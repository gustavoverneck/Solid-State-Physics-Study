# Quantum Mechanics Solvers: From Schrödinger to Dirac

This repository contains a collection of Python scripts for solving quantum mechanical systems, progressing from basic 1D problems to relativistic radial simulations.

## 📂 File Overview

### 1. `1d_schroedinger.py`
**System:** Non-Relativistic Particle in an Infinite Potential Well.
**Method:** Finite Difference Method (3-point stencil).

* **Description:** Solves the time-independent Schrödinger equation in 1D Cartesian coordinates.
* **Key Features:**
    * Discretizes the Hamiltonian $H = -\frac{1}{2}\nabla^2 + V$.
    * Demonstrates basic matrix diagonalization using `numpy`.
    * Validates numerical results against analytical $n^2$ energy levels.

### 2. `hydrogen_atom.py`
**System:** Non-Relativistic Hydrogen Atom (Radial).
**Method:** Finite Difference with Staggered Grid.

* **Description:** Solves the radial Schrödinger equation for the Coulomb potential $V(r) = -1/r$.
* **Key Features:**
    * **Staggered Grid:** Defines the grid starting at $r=h$ (instead of 0) to handle the $1/r$ singularity numerically without crashing.
    * **Effective Potential:** Incorporates the centrifugal barrier $\frac{l(l+1)}{2r^2}$ for handling $s, p, d$ orbitals.
    * Computes expectation values like $\langle r \rangle$.

### 3. `1d_dirac_spectral.py`
**System:** Relativistic Dirac Particle in a Box.
**Method:** Spectral Method (Fast Fourier Transform - FFT).

* **Description:** Solves the 1D Dirac equation for a 2-component spinor.
* **Why FFT?** Naive finite difference methods applied to the Dirac equation result in **Fermion Doubling** (spurious oscillatory states). This script uses FFT to compute derivatives in $k$-space, eliminating these artifacts.
* **Key Features:**
    * **Spectral Momentum:** Constructs a dense Hamiltonian matrix where derivatives are exact.
    * **Soft Walls:** Simulates a box potential using high energy barriers within a periodic FFT domain.
    * **Filtering:** Automatically isolates physical electron states from the Dirac sea and edge artifacts.

### 4. `hydrogen_atom_dirac.py`
**System:** Relativistic Hydrogen Atom (Radial).
**Method:** Coupled Block-Matrix Diagonalization.

* **Description:** Solves the coupled system of first-order radial differential equations for the Dirac spinors (Large component $G$ and Small component $F$).
* **Key Features:**
    * **Block Hamiltonian:** Constructs a $2N \times 2N$ matrix representing the coupling between matter and antimatter/spin components.
    * **Fine Structure:** Accurately reproduces relativistic energy corrections (e.g., the energy difference between $2p_{1/2}$ and $2p_{3/2}$).
    * **Spinors:** Visualizes the "Small Component" ($F$), validating relativistic effects near the nucleus.

## ⚙️ Usage

Dependencies required:
```bash
pip install numpy scipy matplotlib
```

## 📝 Units & Constants

All simulations in this repository are performed using **Atomic Units (Hartree)**. This system normalizes fundamental constants to unity to simplify the numerical implementation.

* **Length:** Bohr radius ($a_0 = 1 \text{ a.u.} \approx 0.529 \AA$)
* **Energy:** Hartree ($E_h = 1 \text{ a.u.} \approx 27.211386 \text{ eV}$)
* **Mass:** Electron mass ($m_e = 1 \text{ a.u.}$)
* **Charge:** Elementary charge ($e = 1 \text{ a.u.}$)
* **Action / Angular Momentum:** Reduced Planck constant ($\hbar = 1 \text{ a.u.}$)
* **Speed of Light:** Inverse fine-structure constant ($c = 1/\alpha \approx 137.035999 \text{ a.u.}$)
