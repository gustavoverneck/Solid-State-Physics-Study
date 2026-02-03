import matplotlib.pyplot as plt
import numpy as np

# --- 1. Parâmetros Físicos e Grade ---
# Usaremos Unidades Atômicas (a.u.)
c = 137.036      # Velocidade da luz
m = 1.0          # Massa do elétron
L = 1.0          # Largura do poço
N_POINTS = 500   # Pontos da grade (Diminui um pouco pois a matriz será 2x maior)

# Passo da grade
h = L / N_POINTS
x_values = np.linspace(0, L, N_POINTS)

# --- 2. Construção do Hamiltoniano de Dirac ---
# O tamanho agora é 2 * N (Componente A e Componente B para cada ponto)
dim = 2 * N_POINTS
H = np.zeros((dim, dim), dtype=np.complex128)

# Constantes auxiliares
# Termo de massa (Energia de repouso)
mass_term = m * c**2

# Termo cinético (Momentum): -i * c * (d/dx)
# Usamos diferenças finitas centrais: d/dx ~ (f(x+h) - f(x-h)) / 2h
# O acoplamento é entre a componente A e B.
kinetic_factor = -1j * c / (2 * h)

# Preenchimento da Matriz por Blocos
# Estrutura: |  mc^2 + V      c*p      |
#            |    c*p        -mc^2 + V |

# --- Blocos Diagonais (Massa + Potencial) ---
# O potencial V=0 dentro do poço, então só temos a massa.
# Bloco Superior Esquerdo (0 a N): +mc^2
for i in range(N_POINTS):
    H[i, i] = mass_term 

# Bloco Inferior Direito (N a 2N): -mc^2
for i in range(N_POINTS, dim):
    H[i, i] = -mass_term

# --- Blocos Fora da Diagonal (Momento/Cinética) ---
# O momento conecta a parte superior (A) com a inferior (B) e vice-versa.
# H_AB = -i * c * d/dx
# H[i, j_lower] onde j_lower é o índice correspondente no bloco de baixo

for i in range(N_POINTS):
    # Índices no bloco inferior (spinor B)
    idx_lower = i + N_POINTS 
    
    # Vizinho à direita (i+1)
    if i < N_POINTS - 1:
        # A acoplando com B(i+1)
        H[i, idx_lower + 1] = kinetic_factor
        # B acoplando com A(i+1) (Hermitiano conjugado)
        H[idx_lower, i + 1] = np.conj(kinetic_factor)
        
    # Vizinho à esquerda (i-1)
    if i > 0:
        # A acoplando com B(i-1) - Sinal trocado pela derivada central
        H[i, idx_lower - 1] = -kinetic_factor
        # B acoplando com A(i-1)
        H[idx_lower, i - 1] = np.conj(-kinetic_factor)

# --- 3. Diagonalização ---
# eigh para matrizes Hermitianas
print("Diagonalizando...")
eigenvalues, eigenvecs = np.linalg.eigh(H)

# --- 4. Análise e Plotagem ---

# Dirac gera estados de energia positiva (elétron) e negativa (pósitron)
# Vamos filtrar apenas os estados de energia positiva (> 0)
positive_indices = np.where(eigenvalues > 0)[0]
energies = eigenvalues[positive_indices]
vectors = eigenvecs[:, positive_indices]

print("\n{:-^50}".format(" RESULTADOS DIRAC (Poço Infinito) "))
print(f"{'n':<5} | {'Energia Total (Ha)':<20} | {'E_cinética (E - mc^2)':<20}")
print("-" * 50)

for i in range(5):
    E_total = energies[i]
    # Subtraímos a energia de repouso para comparar com Schrödinger
    E_kin = E_total - mass_term 
    print(f"{i+1:<5} | {E_total:<20.5f} | {E_kin:<20.5f}")

# Plotagem
plt.figure(figsize=(12, 6))

# Plotar as densidades de probabilidade dos 3 primeiros estados
for i in range(3):
    psi = vectors[:, i]
    
    # Separar componentes A (superior) e B (inferior)
    psi_A = psi[:N_POINTS]
    psi_B = psi[N_POINTS:]
    
    # Densidade de Probabilidade = |A|^2 + |B|^2
    prob_density = np.abs(psi_A)**2 + np.abs(psi_B)**2
    
    # Normalização para o plot
    prob_density = prob_density / (np.sum(prob_density) * h)
    
    # Shift para visualização (empilhar no gráfico)
    shift = energies[i] - mass_term
    
    plt.plot(x_values, prob_density + shift, label=f'n={i+1} (E_kin={shift:.2f})')
    plt.axhline(shift, color='gray', linestyle='--', alpha=0.3)



plt.title("Densidade de Probabilidade - Partícula na Caixa (Dirac)")
plt.xlabel("Posição x (a.u.)")
plt.ylabel("Densidade + Energia Cinética")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()