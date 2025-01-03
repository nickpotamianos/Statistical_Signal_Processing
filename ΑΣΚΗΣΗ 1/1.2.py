import numpy as np

# Θέτουμε το πλήθος των δειγμάτων και το μήκος του εκτιμώμενου FIR
N = 1000
L = 3

# Δημιουργούμε λευκό Gaussian θόρυβο για x(n) με μέση τιμή 0 και διασπορά 1
np.random.seed(0)
x = np.random.normal(0, 1, N)

# Δημιουργούμε την έξοδο d(n) από το πραγματικό FIR:
# d(n) = x(n) - 0.4*x(n-1) - 4*x(n-2) + 0.5*x(n-3)
# με σωστή αντιμετώπιση των αρχικών δειγμάτων (zero-padding)
d = np.zeros(N)
for n in range(N):
    d[n] = x[n]
    if n >= 1:
        d[n] -= 0.4 * x[n - 1]
    if n >= 2:
        d[n] -= 4.0 * x[n - 2]
    if n >= 3:
        d[n] += 0.5 * x[n - 3]

# Δημιουργούμε τον πίνακα X μεγέθους N x (L+1),
# ούτως ώστε η γραμμή n να αντιστοιχεί στο [x(n), x(n-1), x(n-2), x(n-3)]
X = np.zeros((N, L+1))
for n in range(N):
    X[n, 0] = x[n]
    if n >= 1:
        X[n, 1] = x[n-1]
    if n >= 2:
        X[n, 2] = x[n-2]
    if n >= 3:
        X[n, 3] = x[n-3]

# Εφαρμόζουμε τον τύπο Wiener–Hopf με Least Squares:
# w0 = (X^T X)^(-1) X^T d
w0 = np.linalg.inv(X.T @ X) @ (X.T @ d)

# Εμφάνιση των τελικών εκτιμώμενων συντελεστών
print("Τελικοί εκτιμώμενοι συντελεστές Wiener:")
for i, coef in enumerate(w0):
    print(f"w0[{i}] = {coef:.6f}")

# Προαιρετικά, σύγκριση με τους πραγματικούς συντελεστές του H(z)
real_coeffs = np.array([1.0, -0.4, -4.0, 0.5])
print("\nΠραγματικοί συντελεστές:")
for i, rc in enumerate(real_coeffs):
    print(f"h[{i}] = {rc}")
