import numpy as np
import matplotlib.pyplot as plt

# === 1. Βασικές Παράμετροι ===
N = 1000  # Πλήθος δειγμάτων
L = 3  # Τάξη φίλτρου (4 συντελεστές: w0, w1, w2, w3)
mu = 0.01  # Σταθερό βήμα εκμάθησης LMS
num_realizations = 20  # Πλήθος υλοποιήσεων


# === 2. Ορισμός της χρονικά μεταβαλλόμενης κρουστικής απόκρισης ===
# Για παράδειγμα, επιλέγουμε την ομαλή μεταβολή b(n):
def b_smooth(n):
    return 1.0 / (1.0 + np.exp(-0.02 * n))

def b_abrupt(n):
    # Ακαριαία μεταβολή:
    #  b(n) = 100 για 1 <= n <= 500
    #         0   για 501 <= n <= 1000
    if 1 <= n <= 500:
        return 100.0
    else:
        return 0.0


def lms_time_varying_mse(N, L, mu, seed):
    """
    Εκτελεί τον αλγόριθμο LMS για μία υλοποίηση λευκού θορύβου (seed),
    σε ένα χρονικά μεταβαλλόμενο σύστημα με b(n) = b_smooth(n).
    Επιστρέφει το διάνυσμα e^2(n), n=0..N-1.
    """
    # 1. Δημιουργούμε την είσοδο x(n) με το συγκεκριμένο seed
    np.random.seed(seed)
    x = np.random.normal(0, 1, N)

    # 2. Αρχικοποιούμε το FIR φίλτρο και τις δομές δεδομένων
    w_current = np.zeros(L + 1)  # w0, w1, w2, w3
    e_vec = np.zeros(N)

    # 3. Εκτελούμε LMS σε κάθε χρονική στιγμή n
    for n in range(N):
        # Υπολογισμός του d(n) από το χρονικά μεταβαλλόμενο σύστημα
        b_n = b_smooth(n)  # ομαλή μεταβολή
        d_n = b_n * x[n]
        if n >= 1: d_n -= 0.4 * x[n - 1]
        if n >= 2: d_n -= 4.0 * x[n - 2]
        if n >= 3: d_n += 0.5 * x[n - 3]

        # Δημιουργία διανύσματος εισόδου x_vec
        x_vec = np.zeros(L + 1)
        x_vec[0] = x[n]
        if n >= 1: x_vec[1] = x[n - 1]
        if n >= 2: x_vec[2] = x[n - 2]
        if n >= 3: x_vec[3] = x[n - 3]

        # Έξοδος φίλτρου και σφάλμα
        y_n = np.dot(w_current, x_vec)
        e_n = d_n - y_n
        e_vec[n] = e_n

        # Ενημέρωση βαρών
        w_current = w_current + mu * e_n * x_vec

    # Επιστροφή του e^2(n)
    return e_vec ** 2


# === 3. Εκτέλεση πολλαπλών υλοποιήσεων και υπολογισμός μέσου όρου e^2(n) ===

# Πίνακας για την αποθήκευση του e^2(n) σε κάθε υλοποίηση:
all_mse = np.zeros((N, num_realizations))

for r in range(num_realizations):
    # Για κάθε υλοποίηση (r), αλλάζουμε το seed ώστε να πάρουμε διαφορετικό θόρυβο x(n)
    seed = 100 + r  # πχ, seed=100,101,... για 20 υλοποιήσεις
    e2_n = lms_time_varying_mse(N, L, mu, seed)
    all_mse[:, r] = e2_n

# Υπολογισμός του μέσου όρου (over realizations) για κάθε n:
mean_mse = np.mean(all_mse, axis=1)

# === 4. Σχεδίαση της μέσης καμπύλης e^2(n) ===
plt.figure(figsize=(7, 5))
plt.plot(mean_mse, label='Μέσος όρος e^2(n) σε 20 υλοποιήσεις')
plt.title('Καμπύλη μάθησης (LMS) για 20 διαφορετικές υλοποιήσεις, ομαλή μεταβολή b(n)')
plt.xlabel('n')
plt.ylabel('Μέσο τετραγωνικό σφάλμα (e^2(n))')
plt.grid(True)
plt.legend()
plt.show()

# (Προαιρετικά) μπορούμε να τυπώσουμε και τη μέση τιμή των τελευταίων δειγμάτων
avg_steady_state = np.mean(mean_mse[-100:])
print(f"Μέσο Steady-State MSE (τελευταία 100 δείγματα) σε 20 υλοποιήσεις: {avg_steady_state:.6f}")
