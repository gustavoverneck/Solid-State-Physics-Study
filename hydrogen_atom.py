import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. CONFIGURAÇÃO E PARÂMETROS FÍSICOS
# ==========================================
l = 0           # Momento Angular (0=s, 1=p, 2=d)
N_POINTS = 2000 # Aumentei para maior precisão nas integrais
r_max = 60.0    # Tamanho da caixa (em raios de Bohr)

# Grelha Escalonada (Staggered Grid)
# O primeiro ponto é h, não 0, para evitar divisão por zero no potencial
h = r_max / N_POINTS
r = np.linspace(h, r_max, N_POINTS)

def V_eff(r, l):
    """Potencial Efetivo: Coulomb + Barreira Centrífuga"""
    return -1.0/r + l*(l+1)/(2*r**2)

# ==========================================
# 2. CONSTRUÇÃO DO HAMILTONIANO
# ==========================================
# Termo de Energia Cinética: -1/2 * d^2/dr^2
T = -0.5 / h**2

H = np.zeros((N_POINTS, N_POINTS))

# Diagonal Principal: -2*T + V_eff
# (O termo cinético discretizado contribui com -2*T para a diagonal)
diag_kinetic = -2.0 * T 
H[np.diag_indices(N_POINTS)] = diag_kinetic + V_eff(r, l)

# Fora da Diagonal (Hopping terms): T
off_diag = T * np.ones(N_POINTS - 1)
H += np.diag(off_diag, k=1)
H += np.diag(off_diag, k=-1)

# ==========================================
# 3. DIAGONALIZAÇÃO
# ==========================================
print("Diagonalizando matriz Hamiltoniana...")
# eigh é otimizado para matrizes hermitianas (simétricas reais)
eigenvalues, eigenvecs = np.linalg.eigh(H)

# ==========================================
# 4. ANÁLISE DOS RESULTADOS
# ==========================================
print(f"\n{'='*60}")
print(f"ANÁLISE DO ÁTOMO DE HIDROGÊNIO (l={l})")
print(f"{'='*60}")

# Identificar estados ligados (Energia < 0)
bound_indices = [i for i, E in enumerate(eigenvalues) if E < 0]

# Tabela de Energias
print(f"\n{'n':<3} | {'Energia (Ha)':<12} | {'Teórico':<10} | {'Erro (%)':<10} | {'<r> (a.u.)':<10}")
print("-" * 60)

for idx, i in enumerate(bound_indices[:5]): # Analisar os 5 primeiros estados
    E_calc = eigenvalues[i]
    psi_raw = eigenvecs[:, i]
    
    # --- Normalização ---
    # Integral(|u|^2) dr = 1  =>  soma(|u|^2 * h) = 1
    norm = np.sqrt(np.sum(psi_raw**2) * h)
    u = psi_raw / norm
    
    # --- Parâmetros Físicos ---
    n_principal = idx + 1 + l
    prob_density = u**2
    
    # 1. Raio Médio <r> = Integral(r * |u|^2 dr)
    r_medio = np.sum(r * prob_density) * h
    
    # 2. Teórico e Erro
    E_theo = -0.5 / (n_principal**2)
    erro_pct = abs((E_calc - E_theo) / E_theo) * 100
    
    print(f"{n_principal:<3} | {E_calc:<12.5f} | {E_theo:<10.5f} | {erro_pct:<10.4f} | {r_medio:<10.4f}")

# ==========================================
# 5. VERIFICAÇÃO DO TEOREMA DO VIRIAL (Estado Fundamental)
# ==========================================
idx_fund = bound_indices[0]
u_fund = eigenvecs[:, idx_fund] / np.sqrt(np.sum(eigenvecs[:, idx_fund]**2) * h)
E_fund = eigenvalues[idx_fund]

# <V> = Integral(u * V * u)
V_pot_array = V_eff(r, l)
V_esperado = np.sum(u_fund * V_pot_array * u_fund) * h

# <T> = E_total - <V>
T_esperado = E_fund - V_esperado

print(f"\n--- Verificação do Virial (Estado Fundamental) ---")
print(f"Energia Total (E):     {E_fund:.5f} Ha")
print(f"Energia Potencial <V>: {V_esperado:.5f} Ha (Deveria ser 2*E = {2*E_fund:.5f})")
print(f"Energia Cinética <T>:  {T_esperado:.5f} Ha (Deveria ser -E = {-E_fund:.5f})")
print(f"Razão |-<V> / 2<T>|:   {abs(V_esperado / (2*T_esperado)):.5f} (Ideal = 1.00000)")


# ==========================================
# 6. VISUALIZAÇÃO GRÁFICA
# ==========================================
plt.figure(figsize=(12, 6))

# Plotar Densidade de Probabilidade Radial (|u|^2)
for idx in range(3):
    i = bound_indices[idx]
    n_p = idx + 1 + l
    E = eigenvalues[i]
    
    psi = eigenvecs[:, i]
    norm = np.sqrt(np.sum(psi**2) * h)
    u = psi / norm
    
    plt.plot(r, u**2, label=f'n={n_p} (Densidade Radial)')
    
    # Marcar o Raio Médio no gráfico
    r_mean = np.sum(r * u**2) * h
    plt.axvline(r_mean, linestyle=':', alpha=0.3, color='black')



plt.title(f"Densidade de Probabilidade Radial do Hidrogênio (l={l})")
plt.xlabel("Distância do Núcleo (Raios de Bohr)")
plt.ylabel("Densidade de Probabilidade $|u(r)|^2$")
plt.xlim(0, 25)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()