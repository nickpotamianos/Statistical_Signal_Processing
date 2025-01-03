import numpy as np

# 1. Παράμετροι
N = 1000  # Πλήθος δειγμάτων
L = 3  # Τάξη φίλτρου (4 συντελεστές)
np.random.seed(0)  # Για αναπαραγωγιμότητα (προαιρετικά)

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


# 4. Συνάρτηση υλοποίησης LMS, η οποία κρατά την εξέλιξη των βαρών w(n)
def lms_filter(x, d, mu, N, L):
    """
    Εκτελεί τον αλγόριθμο LMS για δεδομένα x(n), d(n),
    βήμα mu, τάξη L (άρα L+1 βάρη) και μήκος N.
    Επιστρέφει τον πίνακα w_evolution (N x (L+1)) με την εξέλιξη των βαρών.
    """
    w_evolution = np.zeros((N, L + 1))  # αποθήκευση των βαρών σε κάθε χρονική στιγμή
    w_current = np.zeros(L + 1)  # αρχικοποίηση με μηδέν

    for n in range(N):
        # Δημιουργούμε το διάνυσμα εισόδου x_vec(n)
        x_vec = np.zeros(L + 1)
        x_vec[0] = x[n]
        if n >= 1: x_vec[1] = x[n - 1]
        if n >= 2: x_vec[2] = x[n - 2]
        if n >= 3: x_vec[3] = x[n - 3]

        # Υπολογισμός εξόδου y(n)
        y = np.dot(w_current, x_vec)

        # Σφάλμα e(n)
        e = d[n] - y

        # Ενημέρωση βαρών
        w_current = w_current + mu * e * x_vec

        # Αποθήκευση τρεχόντων βαρών
        w_evolution[n, :] = w_current

    return w_evolution


# 5. Κύριος βρόχος για τις διάφορες τιμές βήματος
for mu in mu_values:
    # Εκτέλεση LMS
    w_evol = lms_filter(x, d, mu, N, L)

    # Εκτύπωση κεφαλίδας
    print(f"\n=== Αποτελέσματα για μ = {mu:.4f} ===")
    # Προαιρετικά, εκτυπώνουμε την τελική τιμή των βαρών
    final_w = w_evol[-1, :]
    print("Τελικοί συντελεστές:")
    for i, coef in enumerate(final_w):
        print(f"w[{i}] = {coef:.6f}")

    # Προαιρετικά, εκτυπώνουμε επιλεγμένα στιγμιότυπα κατά την εξέλιξη
    # (π.χ. στην αρχή, ενδιάμεσα και στο τέλος)
    snapshots = [0, 1, 2, 10, 50, 100, 500, 999]  # τυχαία επιλεγμένα δείγματα
    print("\nΕξέλιξη συντελεστών w(n) σε επιλεγμένα n:")
    for idx in snapshots:
        w_vec = w_evol[idx, :]
        print(f"n={idx:3d} -> [w0={w_vec[0]:.4f}, w1={w_vec[1]:.4f}, w2={w_vec[2]:.4f}, w3={w_vec[3]:.4f}]")

# 6. (Προαιρετικά) Σύγκριση με πραγματικούς συντελεστές και Wiener
print("\nΠραγματικοί συντελεστές H(z) = [1, -0.4, -4.0, 0.5]")
