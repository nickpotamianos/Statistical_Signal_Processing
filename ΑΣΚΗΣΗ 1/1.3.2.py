import numpy as np

# 1. Παράμετροι
N = 1000          # Πλήθος δειγμάτων
L = 3             # Τάξη φίλτρου (άρα 4 συντελεστές: w0, w1, w2, w3)
np.random.seed(0) # Για αναπαραγωγιμότητα (προαιρετικά)

# 2. Δημιουργία εισόδου x(n): λευκός Gaussian θόρυβος (mean=0, var=1)
x = np.random.normal(0, 1, N)

# 3. Δημιουργία εξόδου d(n) από το πραγματικό σύστημα H(z)
#    d(n) = x(n) - 0.4*x(n-1) - 4*x(n-2) + 0.5*x(n-3)
d = np.zeros(N)
for n in range(N):
    d[n] = x[n]
    if n >= 1:
        d[n] -= 0.4 * x[n-1]
    if n >= 2:
        d[n] -= 4.0 * x[n-2]
    if n >= 3:
        d[n] += 0.5 * x[n-3]

# 4. Ορισμός του LMS
#    Μέγεθος βήματος mu (με βάση μ_max = 2, αφού λευκό x(n) με διασπορά 1 -> λmax=1)
mu_max = 2.0
mu = 0.1 * mu_max  # => mu = 0.2

# 5. Αρχικοποίηση των βαρών w(n) = [w0, w1, w2, w3]^T στο μηδέν
w = np.zeros(L+1)

# 6. Βρόχος LMS
for n in range(N):
    # Δημιουργούμε το διάνυσμα εισόδου x_vec(n) = [x(n), x(n-1), x(n-2), x(n-3)]
    x_vec = np.zeros(L+1)
    x_vec[0] = x[n]
    if n >= 1: x_vec[1] = x[n-1]
    if n >= 2: x_vec[2] = x[n-2]
    if n >= 3: x_vec[3] = x[n-3]

    # Υπολογισμός εξόδου y(n) του φίλτρου w(n)
    y = np.dot(w, x_vec)

    # Υπολογισμός σφάλματος e(n) = d(n) - y(n)
    e = d[n] - y

    # Ενημέρωση του διάνυσματος βαρών
    w = w + mu * e * x_vec

# 7. Εκτυπώνουμε τους τελικούς προσαρμοσμένους συντελεστές LMS
print("Τελικοί συντελεστές μέσω LMS:")
for i, coef in enumerate(w):
    print(f"w[{i}] = {coef:.6f}")

# 8. Σύγκριση με το βέλτιστο φίλτρο από το Ερώτημα 1.2 (Wiener):
#    Εδώ υποθέτουμε ότι το w_opt προκύπτει από την προηγούμενη εκτίμηση Least Squares
#    ή ότι έχει ήδη υπολογιστεί / δοθεί. Για παράδειγμα:
# (Προαιρετικό) Υπολογισμός w_opt (Wiener) με μέθοδο Least Squares για αναφορά
X = np.zeros((N, L+1))
for n in range(N):
    X[n, 0] = x[n]
    if n >= 1: X[n, 1] = x[n-1]
    if n >= 2: X[n, 2] = x[n-2]
    if n >= 3: X[n, 3] = x[n-3]
w_opt = np.linalg.inv(X.T @ X) @ (X.T @ d)

print("\nΣυντελεστές φίλτρου Wiener (από Ερώτημα 1.2):")
for i, coef in enumerate(w_opt):
    print(f"w_opt[{i}] = {coef:.6f}")

# 9. Προαιρετική εκτύπωση των πραγματικών συντελεστών του H(z)
real_coeffs = np.array([1.0, -0.4, -4.0, 0.5])
print("\nΠραγματικοί συντελεστές H(z):")
for i, rc in enumerate(real_coeffs):
    print(f"h[{i}] = {rc}")
