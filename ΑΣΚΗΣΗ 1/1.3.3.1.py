import numpy as np
import matplotlib.pyplot as plt

# 1. Παράμετροι προσομοίωσης
N = 1000          # Πλήθος δειγμάτων
mu = 0.01         # Σταθερό βήμα εκμάθησης
np.random.seed(0) # Για αναπαραγωγιμότητα (προαιρετικά)

# 2. Δημιουργία σήματος εισόδου x(n) και εξόδου d(n)
x = np.random.normal(0, 1, N)
d = np.zeros(N)
for n in range(N):
    d[n] = x[n]
    if n >= 1: d[n] -= 0.4 * x[n-1]
    if n >= 2: d[n] -= 4.0 * x[n-2]
    if n >= 3: d[n] += 0.5 * x[n-3]

# 3. Συνάρτηση LMS που επιστρέφει την ακολουθία του σφάλματος e(n)
def lms_learning_curve(x, d, mu, L):
    """
    Εκτελεί τον αλγόριθμο LMS με FIR τάξης L, σε σήματα x και d,
    και επιστρέφει τον πίνακα e(n) (σφάλμα) για n=0,...,N-1.
    """
    N = len(x)
    w = np.zeros(L+1)    # αρχικοποίηση βαρών
    e_vec = np.zeros(N)  # αποθήκευση του σφάλματος σε κάθε βήμα

    for n in range(N):
        # x_vec: [ x(n), x(n-1), ..., x(n-L) ]
        x_vec = np.zeros(L+1)
        for k in range(L+1):
            if n-k >= 0:
                x_vec[k] = x[n-k]
            else:
                x_vec[k] = 0.0

        y = np.dot(w, x_vec)        # έξοδος εκτιμώμενου φίλτρου
        e = d[n] - y                # σφάλμα e(n)
        e_vec[n] = e

        # ενημέρωση βαρών LMS
        w = w + mu * e * x_vec

    return e_vec

# 4. Εκτέλεση LMS για 3 συντελεστές (L=2) και 5 συντελεστές (L=4)
e_3 = lms_learning_curve(x, d, mu, L=2)   # 3 συντελεστές
e_5 = lms_learning_curve(x, d, mu, L=4)   # 5 συντελεστές

# 5. Καμπύλη μάθησης: μέσο τετραγωνικό σφάλμα σε συνάρτηση του n
mse_3 = e_3**2        # αν θέλουμε στιγμιαία αποτίμηση, ή μπορούμε να πάρουμε έναν κινούμενο μέσο
mse_5 = e_5**2

# (Προαιρετικά) μπορεί να θέλουμε λειασμένο MSE, π.χ. με ολίγον moving average
# ωστόσο εδώ το κρατάμε ως έχει για απλοποίηση.

# 6. Γράφημα MSE έναντι του n
plt.figure(figsize=(8,5))
plt.plot(mse_3, label='ΜSE με 3 συντελεστές (L=2)')
plt.plot(mse_5, label='ΜSE με 5 συντελεστές (L=4)')
plt.title('Καμπύλη Μάθησης LMS (μ = 0.01)')
plt.xlabel('n')
plt.ylabel('Στιγμιαίο Τετραγωνικό Σφάλμα e^2(n)')
plt.legend()
plt.grid(True)
plt.show()

# Προαιρετικά, εκτύπωση των τελικών μέσων τιμών MSE
print(f"Τελικό MSE (Μέσος όρος τελευταίων 100 δειγμάτων) για 3 συντελεστές: {np.mean(mse_3[-100:]):.6f}")
print(f"Τελικό MSE (Μέσος όρος τελευταίων 100 δειγμάτων) για 5 συντελεστές: {np.mean(mse_5[-100:]):.6f}")
