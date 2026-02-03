import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# ==========================================
# 1. PARÂMETROS FÍSICOS
# ==========================================
# Unidades Atômicas
c = 137.035999   # Velocidade da luz (1/alpha)
m = 1.0          # Massa do elétron
Z = 1.0          # Carga nuclear (Hidrogênio)
kappa = -1       # kappa = -1 corresponde ao orbital 1s (j=1/2, l=0)

# ==========================================
# 2. GRID RADIAL
# ==========================================
N = 3000         # Alta resolução necessária para Dirac
r_max = 40.0     # Raio máximo (u.a.)

# Grid escalonado (começa em h, não em 0, para evitar 1/r infinito)
h = r_max / N
r = np.linspace(h, r_max, N)

# Potencial de Coulomb
V = -Z / r

# ==========================================
# 3. OPERADORES DIFERENCIAIS
# ==========================================
# Matriz derivada d/dr usando diferenças finitas centrais (f(i+1) - f(i-1))/2h
# É uma matriz anti-simétrica
ones = np.ones(N-1)
D = (np.diag(ones, k=1) - np.diag(ones, k=-1)) / (2*h)

# Tratamento das bordas da derivada (Forward/Backward simples)
D[0, 1] = 1/(2*h);  # Ajuste simples para não perder conexão
D[-1, -2] = -1/(2*h)

# Operador Kappa/r
K_over_r = np.diag(kappa / r)

# ==========================================
# 4. MONTAGEM DO HAMILTONIANO DE DIRAC (2N x 2N)
# ==========================================
# H |psi> = E |psi>, onde |psi> = [G, F]^T
#
# A estrutura matricial baseada nas equações acopladas:
# H = |  mc^2 + V          c(-d/dr + k/r) |
#     |  c(d/dr + k/r)     -mc^2 + V      |

Identity = np.eye(N)

# Blocos Diagonais
H_GG = (m * c**2) * Identity + np.diag(V)
H_FF = (-m * c**2) * Identity + np.diag(V)

# Blocos Fora da Diagonal (Acoplamento Cinético)
# Note os sinais da derivada baseados na equação teórica
H_GF = c * (-D + K_over_r)
H_FG = c * (D + K_over_r)

# Montar a Matriz Gigante
H = np.block([[H_GG, H_GF], 
              [H_FG, H_FF]])

# ==========================================
# 5. DIAGONALIZAÇÃO
# ==========================================
print("Diagonalizando Matriz Dirac ({}x{})...".format(2*N, 2*N))
# Solicitamos apenas uma faixa de autovalores para economizar tempo
# Queremos estados ligados: Energia < mc^2
# Mas > 0 (apenas para não pegar o mar de Dirac profundo, embora bound states sejam E < mc^2)
eigvals, eigvecs = eigh(H, subset_by_value=[m*c**2 - 2.0, m*c**2 - 0.001])

# ==========================================
# 6. ANÁLISE E RESULTADOS
# ==========================================
print(f"\n{' RESULTADOS DIRAC (H-ATOM) ':-^50}")

# Fórmula Exata de Dirac para Energia (Sommerfeld)
gamma = np.sqrt(kappa**2 - (Z/c)**2)
n_principal = 1 # Para o estado fundamental
# A fórmula abaixo é para E_total (incluindo massa)
E_exact_total = m*c**2 / np.sqrt(1 + (Z * (1/c) / (n_principal - abs(kappa) + gamma))**2)
E_exact_bind = E_exact_total - m*c**2

# Pegar o estado fundamental (menor energia encontrada acima do mar de Dirac)
# O estado fundamental é o primeiro retornado pelo subset_by_value
E_calc_total = eigvals[0]
E_calc_bind = E_calc_total - m*c**2

print(f"Energia de Repouso (mc^2): {m*c**2:.6f} Ha")
print(f"Energia Total Calc:        {E_calc_total:.6f} Ha")
print("-" * 50)
print(f"Energia de Ligação (Calc):  {E_calc_bind:.9f} Ha")
print(f"Energia de Ligação (Exata): {E_exact_bind:.9f} Ha")
print(f"Erro: {abs(E_calc_bind - E_exact_bind):.2e} Ha")
print(f"Comparação Schrödinger:     -0.500000000 Ha")

# ==========================================
# 7. PLOTAGEM (Spinors)
# ==========================================
psi = eigvecs[:, 0]

# Separar Componente Grande (G) e Pequena (F)
G = psi[:N]
F = psi[N:]

# Normalização: Integral(|G|^2 + |F|^2) dr = 1
norm = np.sqrt(np.sum(G**2 + F**2) * h)
G = G / norm
F = F / norm

plt.figure(figsize=(10, 6))

# Plotar Densidades
plt.plot(r, G, 'b-', label='Componente Grande G(r) (Matéria)', linewidth=2)
plt.plot(r, F, 'r-', label='Componente Pequena F(r) (Relativística)', linewidth=2)
plt.plot(r, G**2 + F**2, 'k--', label='Densidade Total $|G|^2+|F|^2$', alpha=0.3)

plt.title(f"Funções de Onda Radiais de Dirac (1s) - $\kappa={kappa}$")
plt.xlabel("Distância (u.a.)")
plt.ylabel("Amplitude")
plt.xlim(0, 10)
plt.legend()
plt.grid(True, alpha=0.3)

# Inserir texto com a razão entre as componentes
ratio = np.max(np.abs(F)) / np.max(np.abs(G))
plt.text(2, 0.2, f"Razão F/G $\\approx$ {ratio:.4f}\n($\sim Z\\alpha/2$)", bbox=dict(facecolor='white', alpha=0.8))

plt.show()