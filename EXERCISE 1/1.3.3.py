import numpy as np
import matplotlib.pyplot as plt

# 1. Παράμετροι
N = 1000  # Πλήθος δειγμάτων
L = 3  # Τάξη φίλτρου (4 συντελεστές)
np.random.seed(0)  # Για αναπαραγωγιμότητα

# 2. Δημιουργία εισόδου x(n) και επιθυμητού d(n)
x = np.random.normal(0, 1, N)
d = np.zeros(N)
for n in range(N):
    d[n] = x[n]
    if n >= 1: d[n] -= 0.4 * x[n - 1]
    if n >= 2: d[n] -= 4.0 * x[n - 2]
    if n >= 3: d[n] += 0.5 * x[n - 3]

# 3. Ορισμός των διαφορετικών τιμών βήματος
mu_max = 2.0  # (για λευκό x(n) με διασπορά 1, λmax = 1)
mu_values = [0.001 * mu_max, 0.01 * mu_max, 0.1 * mu_max, 0.5 * mu_max]


# 4. Συνάρτηση υλοποίησης LMS
def lms_filter(x, d, mu, N, L):
    w_evolution = np.zeros((N, L + 1))
    w_current = np.zeros(L + 1)
    error = np.zeros(N)  # Προσθήκη για αποθήκευση σφάλματος

    for n in range(N):
        x_vec = np.zeros(L + 1)
        x_vec[0] = x[n]
        if n >= 1: x_vec[1] = x[n - 1]
        if n >= 2: x_vec[2] = x[n - 2]
        if n >= 3: x_vec[3] = x[n - 3]

        y = np.dot(w_current, x_vec)
        e = d[n] - y
        error[n] = e  # Αποθήκευση σφάλματος
        w_current = w_current + mu * e * x_vec
        w_evolution[n, :] = w_current

    return w_evolution, error


# 5. Δημιουργία subplots για κάθε τιμή του μ
# Use a built-in style
plt.style.use('default')
fig, axes = plt.subplots(len(mu_values), 2, figsize=(15, 4 * len(mu_values)))
fig.suptitle('Εξέλιξη συντελεστών LMS και σφάλματος για διάφορες τιμές του μ', fontsize=16)

# Πραγματικοί συντελεστές για σύγκριση
true_coeffs = np.array([1, -0.4, -4.0, 0.5])

# 6. Κύριος βρόχος για τις διάφορες τιμές βήματος
for idx, mu in enumerate(mu_values):
    # Εκτέλεση LMS
    w_evol, error = lms_filter(x, d, mu, N, L)

    # Plot εξέλιξης συντελεστών
    ax_coef = axes[idx, 0]
    for i in range(L + 1):
        ax_coef.plot(w_evol[:, i], label=f'w{i}')
        ax_coef.axhline(y=true_coeffs[i], color=f'C{i}', linestyle=':', alpha=0.5)

    ax_coef.set_title(f'Εξέλιξη συντελεστών (μ = {mu:.4f})')
    ax_coef.set_xlabel('Επαναλήψεις n')
    ax_coef.set_ylabel('Τιμή συντελεστή')
    ax_coef.grid(True)
    ax_coef.legend()

    # Plot σφάλματος
    ax_error = axes[idx, 1]
    ax_error.plot(error ** 2, label='MSE')
    ax_error.set_title(f'Μέσο τετραγωνικό σφάλμα (μ = {mu:.4f})')
    ax_error.set_xlabel('Επαναλήψεις n')
    ax_error.set_ylabel('MSE')
    ax_error.grid(True)
    ax_error.set_yscale('log')  # Λογαριθμική κλίμακα για καλύτερη απεικόνιση

    # Εκτύπωση τελικών συντελεστών
    print(f"\n=== Αποτελέσματα για μ = {mu:.4f} ===")
    final_w = w_evol[-1, :]
    print("Τελικοί συντελεστές:")
    for i, coef in enumerate(final_w):
        print(f"w[{i}] = {coef:.6f}")

plt.tight_layout()
plt.show()

# 7. Εκτύπωση πραγματικών συντελεστών
print("\nΠραγματικοί συντελεστές H(z) = [1, -0.4, -4.0, 0.5]")