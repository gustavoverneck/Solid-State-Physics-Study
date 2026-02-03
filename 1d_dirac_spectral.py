import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. PARÂMETROS FÍSICOS E GRELHA
# ==========================================
# Unidades Atômicas (Hartree)
c = 137.036       # Velocidade da luz
m = 1.0           # Massa do elétron
L = 1.0           # Tamanho total do domínio simulação
N = 256           # Pontos da grade (Potência de 2 é ideal para FFT)

# Grade espacial (Endpoint=False garante periodicidade perfeita para FFT)
x = np.linspace(0, L, N, endpoint=False)
dx = L / N

# ==========================================
# 2. OPERADOR MOMENTO (p) VIA FFT
# ==========================================
# Construímos a matriz do operador momento no espaço real.
# p = F^-1 * (hbar * k) * F
print("Construindo operador Momento via FFT...")

# Frequências angulares k = 2*pi * f
k_values = np.fft.fftfreq(N, d=dx) * 2 * np.pi

# Criando a matriz densa p (NxN)
p_matrix = np.zeros((N, N), dtype=np.complex128)

for i in range(N):
    # Vetor base (delta em i)
    u = np.zeros(N)
    u[i] = 1.0
    
    # Derivada Espectral: FFT -> Multiplica por k -> IFFT
    # p = -i * d/dx  --> no espaço k vira: +k
    # (Nota: na física p = hbar*k. Em a.u. hbar=1)
    u_k = np.fft.fft(u)
    du_k = u_k * k_values
    p_col = np.fft.ifft(du_k)
    
    p_matrix[:, i] = p_col

# ==========================================
# 3. POTENCIAL E HAMILTONIANO DE DIRAC
# ==========================================
# Criamos uma "Caixa Suave" com paredes altas nas bordas
# Isso define a caixa física dentro do domínio periódico
V = np.zeros(N)
wall_height = 20000.0
edge_width = int(N * 0.1) # 10% de parede em cada lado

V[:edge_width] = wall_height
V[-edge_width:] = wall_height

# Tamanho efetivo da caixa (para comparação teórica)
L_effective = L - (2 * (edge_width * dx))

# Matriz 2N x 2N por blocos
# H = |  mc^2 + V      c * p   |
#     |  c * p       -mc^2 + V |

diag_top = (m * c**2) + V
diag_bot = (-m * c**2) + V

H11 = np.diag(diag_top)
H22 = np.diag(diag_bot)
H12 = c * p_matrix
H21 = c * p_matrix # p é Hermitiano

H = np.block([[H11, H12], 
              [H21, H22]])

# ==========================================
# 4. DIAGONALIZAÇÃO E FILTRAGEM
# ==========================================
print("Diagonalizando matriz densa...")
evals, evecs = np.linalg.eigh(H)

# --- FILTRO INTELIGENTE ---
# Critério 1: Energia > Energia de Repouso (mc^2) -> Elimina pósitrons/mar de Dirac
# Critério 2: Energia < Topo da Parede -> Elimina estados livres/não confinados
# Critério 3: Margem de segurança (+0.1) para evitar estados de borda espúrios

E_min = m * c**2 + 0.1
E_max = m * c**2 + wall_height * 0.9 # Pegar só o que está bem abaixo do topo

physical_indices = np.where((evals > E_min) & (evals < E_max))[0]

# Selecionar apenas os dados filtrados
energies = evals[physical_indices]
vectors = evecs[:, physical_indices]

# ==========================================
# 5. EXIBIÇÃO DOS RESULTADOS
# ==========================================
print(f"\n{'-'*60}")
print(f"{'RESULTADOS FINAIS (DIRAC FFT)':^60}")
print(f"{'-'*60}")
print(f"Largura Efetiva da Caixa (L_eff): {L_effective:.4f} a.u.")
print(f"{'-'*60}")
print(f"{'n':<5} | {'E_cinética (Ha)':<20} | {'Teórico (L_eff)':<20} | {'Erro (%)'}")
print(f"{'-'*60}")

# Analisar os 5 primeiros estados encontrados
for i in range(min(5, len(energies))):
    n_quantico = i + 1
    
    # Energia Cinética Simulada
    E_tot = energies[i]
    E_kin = E_tot - m * c**2
    
    # Energia Teórica (Schrödinger Aprox para L_effective)
    # E = (n^2 * pi^2) / (2 * L^2)
    E_theo = (n_quantico**2 * np.pi**2) / (2 * L_effective**2)
    
    erro = abs((E_kin - E_theo)/E_theo) * 100
    
    print(f"{n_quantico:<5} | {E_kin:<20.5f} | {E_theo:<20.5f} | {erro:.2f}%")

# ==========================================
# 6. PLOTAGEM
# ==========================================
plt.figure(figsize=(12, 7))

# Cores para os estados
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Plotar os 3 primeiros estados físicos
for i in range(min(3, len(energies))):
    n = i + 1
    psi = vectors[:, i]
    
    # Densidade de Probabilidade = |A|^2 + |B|^2
    #psi_A = psi[:N]
    #psi_B = psi[N:]
    densidade = np.abs(psi[:N])**2 + np.abs(psi[N:])**2
    
    # Normalização visual (pelo máximo)
    densidade = densidade / np.max(densidade)
    
    E_kin = energies[i] - m*c**2
    
    # Plot preenchido
    plt.plot(x, densidade, color=colors[i], lw=2, label=f'n={n} (E={E_kin:.2f})')
    plt.fill_between(x, 0, densidade, color=colors[i], alpha=0.15)

# Desenhar as paredes (Visualização esquemática)
plt.axvline(edge_width * dx, color='k', linestyle='--', alpha=0.7)
plt.axvline(L - (edge_width * dx), color='k', linestyle='--', alpha=0.7)
plt.text(0.02, 0.5, "PAREDE", rotation=90, verticalalignment='center', color='gray')
plt.text(L-0.05, 0.5, "PAREDE", rotation=90, verticalalignment='center', color='gray')

plt.title(f"Densidade de Probabilidade - Dirac na Caixa (L efetivo = {L_effective:.1f})")
plt.xlabel("Posição x (a.u.)")
plt.ylabel("Densidade Normalizada")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, L)
plt.ylim(0, 1.1)

plt.show()