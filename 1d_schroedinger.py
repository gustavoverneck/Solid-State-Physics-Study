import matplotlib.pyplot as plt
import numpy as np

def V(x):
    """Infinite potential well (V=0 inside)."""
    return 0.0 * x

N_POINTS = 1000 
X_MIN = 0.0     # x lower limit
X_MAX = 1.0     # x upper limit

h = (X_MAX - X_MIN) / N_POINTS    # grid step


# Kinetic Energy Term: -1/2 * d^2/dx^2
T = -0.5 / h**2
H = np.zeros([N_POINTS, N_POINTS], dtype=np.float64)
x_values = np.linspace(X_MIN, X_MAX, N_POINTS)

for i in range(N_POINTS):
    v_x = V(x_values[i])
    H[i, i] = -2.0 * T + v_x

    if i < N_POINTS - 1:
        H[i, i+1] = T
        H[i+1, i] = T

# Diagonalization
eigenvalues, eigenvecs = np.linalg.eigh(H)

# Results printing
print("\n{:-^40}".format(" RESULTS "))
print(f"{'n':<5} | {'E':<10} | {'Theo':<10} | {'Error':<10}")
print("-" * 40)

L = X_MAX - X_MIN
for i in range(10):
    e = eigenvalues[i]
    quantum_n = i + 1
    expected = (quantum_n**2 * np.pi**2) / (2 * L**2)
    error = abs(e - expected)
    print(f"{i:<5} | {e:<10.5f} | {expected:<10.5f} | {error:.2e}")

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x_values, V(x_values), 'k--', label='Potential V(x)')

for i in range(4):
    # Scale wavefunction for visibility and shift by energy level
    psi = eigenvecs[:, i]
    # Normalize roughly for plot
    psi_normalized = psi / np.sqrt(h) 
    # Use -psi if it comes out negative (arbitrary phase) to match typical textbook look if needed, 
    # but strictly not required.
    plt.plot(x_values, psi_normalized + eigenvalues[i], label=f'n={i+1}')

plt.title("Infinite Potential Well Wavefunctions")
plt.xlabel("x")
plt.ylabel("Energy / $\psi(x)$")
plt.legend()
plt.grid(True)
plt.show()
