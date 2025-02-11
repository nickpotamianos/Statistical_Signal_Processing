# ΑΣΚΗΣΗ 1 (1η Εργαστηριακή Άσκηση 2024-25.pdf)
Πανεπιστήμιο Πατρών  
Τμήμα Μηχ. Η/Υ & Πληροφορικής  

Στατιστική Επεξεργασία Σήματος και Μάθηση  

Πρώτη Εργαστηριακή Άσκηση  
Ακαδημαϊκό Έτος 2024/25  

**Ταυτοποίηση άγνωστου συστήματος (System Identification)**

Στην παρούσα εργασία θα ασχοληθούμε με ένα από τα βασικότερα προβλήματα της στατιστικής επεξεργασίας σημάτων και μάθησης (με πληθώρα εφαρμογών), που είναι η ταυτοποίηση ενός αγνώστου συστήματος (system identification), το οποίο μπορεί, γενικώς, να είναι και χρονικά μεταβαλλόμενο. Η διαδικασία της ταυτοποίησης πραγματοποιείται με την εκτίμηση των συντελεστών του αγνώστου συστήματος.

## 1. Γραμμικό Χρονικά Αμετάβλητο Σύστημα

![Εικόνα 1](https://github.com/nickpotamianos/Statistical_Signal_Processing/blob/main/EXERCISE%201/image1.png)

**Εικόνα 1:** Η διαδικασία ταυτοποίησης γραμμικού χρονικά αμετάβλητου συστήματος.

Για τους σκοπούς αυτής της άσκησης θεωρούμε πως το άγνωστο σύστημα έχει (άγνωστη) πεπερασμένη κρουστική απόκριση h, δηλαδή μπορεί να μοντελοποιηθεί ως ένα FIR φίλτρο, όπως φαίνεται στην Εικόνα 1.

Επίσης, όπως απεικονίζεται στην Εικόνα 1, το σήμα εισόδου x(n)  (το οποίο θεωρήστε πως είναι λευκός Gaussian θόρυβος με μέση τιμή 0 και διασπορά 1 και πως αποτελείται από 1000 δείγματα) εισέρχεται παράλληλα στο άγνωστο σύστημα με συνάρτηση μεταφοράς H(z) , και στο FIR φίλτρο, με συνάρτηση μεταφοράς W(z) . Θεωρούμε ότι η έξοδος d(n) είναι άμεσα παρατηρήσιμη, χωρίς να μολύνεται από θόρυβο παρατήρησης.

Για τους πειραματικούς λόγους της παρούσας άσκησης θεωρήστε πως το άγνωστο σύστημα περιγράφεται από το FIR φίλτρο H(z), με συνάρτηση μεταφοράς

$$
H(z) = 1 - 0.4z^{-1} - 4z^{-2} + 0.5z^{-3}.
$$

Έτσι, η κρουστική απόκριση του υπό αναγνώριση συστήματος έχει την έκφραση,

$$
h(n) = \delta[n] - 0.4\delta[n - 1] - 4\delta[n - 2] + 0.5\delta[n - 3].
$$

Συνεπώς, η έξοδος του υπό αναγνώριση συστήματος θα εξαρτάται από την τρέχουσα τιμή του σήματος εισόδου καθώς και από ένα πεπερασμένο πλήθος παρελθοντικών τιμών του σήματος εισόδου, και πιο συγκεκριμένα θα δίνεται από την σχέση

$$
d(n) = x[n] - 0.4x[n - 1] - 4x[n - 2] + 0.5x[n - 3].
$$

Ομοίως θεωρούμε πως το FIR σύστημα \( W(z) \) έχει πεπερασμένη κρουστική απόκριση, η έξοδος του οποίου θα εξαρτάται από την τρέχουσα τιμή του σήματος εισόδου καθώς και από ένα πεπερασμένο πλήθος παρελθοντικών τιμών του σήματος εισόδου, δηλαδή

$$
y(n) = \sum_{k=0}^{L} w_n,_k x[n - k].
$$

Η τελευταία σχέση μπορεί να γραφεί ισοδύναμα ως ένα εσωτερικό διάνυσμα δύο διανυσμάτων, όπου το ένα διάνυσμα θα είναι το διάνυσμα των συντελεστών του FIR φίλτρου

$$
\mathbf{w}(n) = [w_0(n), w_1(n), \dots, w_L(n)],
$$

και το άλλο διάνυσμα θα περιέχει τα αντίστοιχα τρέχοντα δείγματα του σήματος εισόδου, δηλαδή

$$
\mathbf{x}(n) = [x(n), x(n - 1), \dots, x(n - L)].
$$

Συνεπώς η σχέση που δίνει την έξοδο του FIR φίλτρου είναι

$$
y(n) = \mathbf{w}(n)^T \mathbf{x}(n).
$$

**Ζητούμενα:**

### 1.1 (Θεωρητικό Ερώτημα)
Να διατυπωθεί θεωρητικά το βέλτιστο φίλτρο Wiener w(n).

### 1.2 (Πειραματικό Ερώτημα)
Υπολογίστε, με βάση τα δεδομένα, το φίλτρο Wiener  w(n), για την εκτίμηση των συντελεστών του H(z).

### 1.3 (Θεωρητικό Ερώτημα)
Κάνοντας εφαρμογή του αλγορίθμου LMS, προσπαθήστε να βρείτε ένα άνω φράγμα για την ποσότητα

$$
E[\mathbf{ŵ}(n)] = E[\mathbf{w}(n) - \mathbf{w}_0],
$$

όπου $$\mathbf{w}_0$$ είναι το βέλτιστο διάνυσμα συντελεστών, διασφαλίζοντας ταυτόχρονα ότι ο LMS αλγόριθμος συγκλίνει.

#### 1.3.1 (Θεωρητικό Ερώτημα)
Καθορίστε το διάστημα τιμών που δύναται να λαμβάνει το μέγεθος βήματος $$\mu$$ ώστε ο αλγόριθμος να συγκλίνει υπό την έννοια της μέσης τιμής.

#### 1.3.2 (Πειραματικό Ερώτημα)
Υλοποιήστε τον αλγόριθμο LMS. Χρησιμοποιείστε ένα φίλτρο w(n) με 4 συντελεστές. Αρχικοποιήστε το φίλτρο με μηδενική τιμή και χρησιμοποιήστε βήμα

$$
\mu = 0.1\mu_{\text{max}},
$$

όπου $$\mu_{\text{max}}$$ είναι το άνω φράγμα του βήματος για σύγκλιση ως προς τη μέση τιμή. Συγκρίνετε τους συντελεστές w(n)  με τους συντελεστές που προέκυψαν από το ερώτημα [1.2].

#### 1.3.3 (Πειραματικό Ερώτημα)
Πραγματοποιείστε τον αλγόριθμο LMS για τις πιθανές τιμές του βήματος

$$
\mu = [0.001\mu_{max},\ 0.01\mu_{max},\ 0.1\mu_{max},\ 0.5\mu_{max}].
$$

Τυπώστε την εξέλιξη των συντελεστών για κάθε περίπτωση.

##### 1.3.3.1 (Πειραματικό Ερώτημα)
Τυπώστε την καμπύλη μάθησης για τον αλγόριθμο LMS όταν το βήμα είναι $$\mu = 0.01$$ και το πλήθος των συντελεστών του FIR φίλτρου είναι 3 και 5. Τι παρατηρείτε;

## 2. Γραμμικό Χρονικά Μεταβαλλόμενο Σύστημα

Στο ερώτημα αυτό, μας ενδιαφέρει η περίπτωση στην οποία το σύστημα το οποίο επιθυμούμε να αναγνωρίσουμε είναι χρονικά μεταβαλλόμενο. Θα ασχοληθούμε με δύο διαφορετικές προσεγγίσεις. Στην πρώτη, θα πραγματοποιείται μία ομαλή μεταβολή του των συντελεστών του αγνώστου συστήματος, ενώ στην δεύτερη θα πραγματοποιείται μία πιο ακαριαία μεταβολή των συντελεστών του αγνώστου συστήματος.

![Εικόνα 2](https://github.com/nickpotamianos/Statistical_Signal_Processing/blob/main/EXERCISE%201/image2.png)

**Εικόνα 2:** Η διαδικασία ταυτοποίησης γραμμικού χρονικά μεταβαλλόμενου συστήματος.

Όπως απεικονίζεται στην Εικόνα 2, η μόνη διαφορά είναι πως τώρα έχουμε ένα χρονικά μεταβαλλόμενο σύστημα $$H_n(z)$$. Ας θεωρήσουμε πως το σύστημα αυτό έχει κρουστική απόκριση χρονικά μεταβαλλόμενη.

### 2.1 Ομαλή Μεταβολή

Αρχικά θεωρούμε πως η κρουστική απόκριση υφίσταται μία ομαλή χρονική μεταβολή με τον εξής τρόπο,

$$
h(n) = b(n)\delta[n] - 0.4\delta[n - 1] - 4\delta[n - 2] + 0.5\delta[n - 3],
$$

όπου b(n) είναι ένας χρονικά μεταβαλλόμενος συντελεστής που εξελίσσεται ως εξής,

$$
b(n) = \frac{1}{1 + e^{-0.02n}}, \quad n \in [1, 1000].
$$

### 2.2 Ακαριαία Μεταβολή

Στην συνέχεια θεωρούμε ότι η κρουστική απόκριση ακολουθεί μία πιο ακαριαία χρονική μεταβολή σύμφωνα με τον κανόνα:

$$
h(n) = b(n)\delta[n] - 0.4\delta[n - 1] - 4\delta[n - 2] + 0.5\delta[n - 3],
$$

όπου b(n) είναι ένας χρονικά μεταβαλλόμενος συντελεστής που εξελίσσεται ως εξής,

$$
b(n) =
\begin{cases}
100, & 1 \leq n \leq 500 \\
0, & 500 < n \leq 1000
\end{cases}.
$$

**Ζητούμενα:**

### 2.3 (Πειραματικό Ερώτημα)
Να σχεδιάσετε την καμπύλη μάθησης του αλγορίθμου LMS για κατάλληλη τιμή βήματος και για τις δύο πιθανές κρουστικές αποκρίσεις των ερωτημάτων [2.1] και [2.2].

### 2.4 (Πειραματικό Ερώτημα)
Να επαναλάβετε τις μετρήσεις για 20 διαφορετικές υλοποιήσεις του σήματος αναφοράς d(n). Για κάθε υλοποίηση, υπολογίστε το στιγμιαίο τετραγωνικό σφάλμα $$e^2(n)$$ συναρτήσει του n, και υπολογίστε το μέσο όρο του για όλες τις υλοποιήσεις.

## Διαδικαστικά

- **Μπορείτε να χρησιμοποιήσετε οποιαδήποτε γλώσσα προγραμματισμού επιθυμείτε.**
Πανεπιστήμιο Πατρών

# ΑΣΚΗΣΗ 2 (2η Εργαστηριακή Άσκηση 2024-25.pdf)

Τμήμα Μηχ. Η/Υ & Πληροφορικής

Στατιστική Επεξεργασία Σήματος και Μάθηση

Δεύτερη Εργαστηριακή Άσκηση

Ακαδημαϊκό Έτος 2024/25

## 1. Ομοσπονδιακή Μάθηση (Federated Learning)

Στην παρούσα εργασία θα ασχοληθούμε με την ομοσπονδιακή μάθηση (που είναι τεχνική κατανεμημένης μάθησης).

Κατ' αρχάς πρέπει να σημειωθεί ότι η κεντρικοποιημένη μηχανική μάθηση [Εικόνα 1] αντιμετωπίζει σημαντικές προκλήσεις όταν εφαρμόζεται σε πραγματικές συνθήκες. Ένα κρίσιμο ζήτημα είναι η ασφάλεια των δεδομένων. Η συγκέντρωση μεγάλου όγκου δεδομένων σε ένα κεντρικό σημείο, όπως συμβαίνει συχνά στις εφαρμογές μηχανικής μάθησης, τα καθιστά ευάλωτα σε κυβερνοεπιθέσεις. Τομείς όπως οι υπηρεσίες υγείας και οικονομίας απαιτούν ιδιαίτερα υψηλά επίπεδα ασφαλείας και προστασίας της ιδιωτικότητας. Επιπλέον, η εκπαίδευση πολύπλοκων μοντέλων μηχανικής μάθησης απαιτεί τεράστια υπολογιστική ισχύ και ενέργεια, γεγονός που μπορεί να περιορίσει την εφαρμογή της σε μεγάλη κλίμακα.

[Εικόνα 1: Κεντρικοποιημένη Μηχανική Μάθηση]

Τέλος, πολύπλοκες εφαρμογές μηχανικής μάθησης, όπως η αυτόνομη οδήγηση ή η αυτόνομη επικοινωνία συσκευών στο πεδίο στις παρυφές του δικτύου (Edge Computing), απαιτούν λήψη αποφάσεων σε πραγματικό χρόνο. Η κεντρική επεξεργασία των δεδομένων μπορεί να επιβραδύνει σημαντικά αυτόν τον χρόνο απόκρισης, θέτοντας σε κίνδυνο την αποτελεσματικότητα και την ασφάλεια τέτοιων συστημάτων.

Μία εναλλακτική προσέγγιση σε αυτά τα προβλήματα προσφέρει η κατανεμημένη μηχανική μάθηση, όπως η λεγόμενη Ομοσπονδιακή Μάθηση (Federated Learning) που φαίνεται σχηματικά στην [Εικόνα 2]. Αντί να συγκεντρώνονται όλα τα δεδομένα σε ένα κεντρικό σημείο, η κατανεμημένη μάθηση επιτρέπει την εκπαίδευση μοντέλων μηχανικής μάθησης σε πολλαπλούς υπολογιστικούς κόμβους που βρίσκονται σε διαφορετικά γεωγραφικά σημεία. Με αυτόν τον τρόπο, προστατεύεται η ιδιωτικότητα των δεδομένων, αποφεύγοντας την ανταλλαγή των δεδομένων αυτών καθ'αυτών και μειώνεται ο κίνδυνος παραβίασης της ασφάλειας, μιας και αν δεχθεί επίθεση ένας κόμβος, όλο το υπόλοιπο δίκτυο παραμένει ασφαλές. Επιπλέον, η εκπαίδευση πολύπλοκων μοντέλων μηχανικής μάθησης μπορεί να διευκολυνθεί μιας και πλέον η διαδικασία μάθησης κατανέμεται σε πολλούς κόμβους έχοντας ως απότοκο την επεξεργασία μεγαλύτερων όγκων δεδομένων, με πιο φθηνή υπολογιστική ισχύ χωρίς να απαιτείται ένας υπερσύγχρονος υπολογιστικός εξοπλισμός σε ένα μόνο σημείο.

[Εικόνα 2: Ομοσπονδιακή Μάθηση]

Στην συνέχεια, θα ορίσουμε το πρόβλημα της κεντρικοποιημένης μηχανικής μάθησης (centralized machine learning) και έπειτα θα δούμε πως διαμορφώνεται το πρόβλημα στην κατανεμημένη εκδοχή του.

Θεωρούμε το πρόβλημα ταξινόμησης πολλαπλών κλάσεων, με διανύσματα εισόδου $x \in X \subseteq R^d$ με αντίστοιχα διανύσματα-ετικέτες $y \in Y = \{1,2,\cdots,C\}$, όπου $C$ είναι το σύνολο των κλάσεων. Αρχικά, το πρόβλημα βελτιστοποίησης που θέλουμε να λύσουμε κατά την κεντρικοποιημένη μηχανική μάθηση ορίζεται ως εξής,

$$
\min_{\theta} F(\theta) := E_{(x,y)\sim D} [l(\phi(x;\theta),y)],
$$

όπου $(x,y) \sim D$ είναι τα διαθέσιμα δεδομένα τα οποία ακολουθούν μία κατανομή $D$, με $l(\phi(x;\theta),y)$ συμβολίζουμε τη συνάρτηση κόστους που μετρά την απόσταση που έχει η πρόβλεψη της παραμετρικής συνάρτησης $\phi(\cdot;\theta)$ (όπου με $\theta \in \Theta$ συμβολίζουμε τις παραμέτρους) για την είσοδο $x$, σε σχέση με την πραγματική τιμή $y$. Ιδανικά θέλουμε να γνωρίζουμε τις βέλτιστες παραμέτρους $\theta^*$ για τις οποίες η συνάρτηση $\phi(x;\theta^*)$ αντιστοιχεί την είσοδο $x$ στην επιθυμητή έξοδο $y$. Όταν συμβαίνει αυτό η τιμή της συνάρτησης κόστους είναι ελάχιστη. Με $E[\cdot]: Z \to R$, συμβολίζουμε την αναμενόμενη τιμή, με $Z$ συμβολίζουμε την τυχαία μεταβλητή του προβλήματός μας. Ο τελεστής $E[\cdot]$ εφαρμόζεται διότι τα δεδομένα μας είναι τυχαία (δηλαδή το $Z$ αντικαθίσταται με τα τυχαία δείγματα $(x,y)$).

Για να μπορέσουμε να βρούμε αναλυτικά το Expectation $E[\cdot]$ χρειάζεται να γνωρίζουμε την συνάρτηση πυκνότητας πιθανότητας της κατανομής $D$, ωστόσο αυτό δεν καθίσταται δυνατό. Κάνουμε επομένως, μία εμπειρική εκτίμηση της μέσης τιμής χρησιμοποιώντας τα πειραματικά δεδομένα, θεωρώντας πως έχουμε αρκούντως πολλά δεδομένα (θεωρητικώς άπειρα δεδομένα) για να προσεγγίσουμε την πραγματική μέση τιμή.

Έτσι λοιπόν το πρόβλημα πλέον παίρνει την εμπειρική του μορφή,

$$
\min_{\theta} \sum_{i=1}^N \frac{1}{N} [l(\phi(x_i;\theta),y_i)].
$$

Δηλαδή το Expectation $E[\cdot]$ έχει αντικατασταθεί από εμπειρική μέση τιμή, η οποία για λόγους απλότητας θεωρούμε πως δίνεται με βάρος $\frac{1}{N}$ για κάθε δείγμα, όπου $N$ είναι το πλήθος των δειγμάτων.

Το παραπάνω πρόβλημα μπορεί να λυθεί με κάποιο επαναληπτικό αλγόριθμο πρώτης τάξης, λόγου χάρη με την χρήση του Gradient Descent (GD), ή κάποιας παραλλαγής του. Οι παράμετροι $\theta$ εξελίσσονται ως εξής,

$$
\theta^{t+1} \leftarrow \theta^t - \eta\nabla_{\theta}F(\theta^t),
$$

όπου $t \in [0,T]$ είναι το πλήθος των βημάτων που θα κάνει ο αλγόριθμος μέχρι να συγκλίνει, δηλαδή $\theta^T \to \theta^*$.

Κατά την κατανεμημένη μηχανική μάθηση το πρόβλημα διαφοροποιείται ως εξής:

$$
\min_{\theta} \frac{1}{K}\sum_{i=1}^K F_i(\theta) := \frac{1}{K}\sum_{i=1}^K E_{(x,y)\sim D_i} [l_i(\phi(x;\theta),y)] := \frac{1}{K}\sum_{i=1}^K \frac{1}{N_i}\sum_{j=1}^{N_i} [l_i(\phi(x_j;\theta),y_j)],
$$

όπου $K$ το πλήθος των χρηστών. Ένας ευρύτατα αποδεκτός τρόπος για να λυθεί το πρόβλημα της κατανεμημένης μάθησης είναι με χρήση του αλγορίθμου federated averaging (FedAvg). Πιο αναλυτικά, έστω ένα υποσύνολο από χρήστες $S$ των $K$ χρηστών που ενορχηστρώνονται από έναν Server [Εικόνα 2] για να εκπαιδεύσουν από κοινού ένα μοντέλο μηχανικής μάθησης. Θεωρούμε πως κάθε χρήστης διαθέτει ένα ιδιωτικό σύνολο δεδομένων αποτελούμενο από $N_i$ δείγματα (ζεύγη $(x,y)$). Στην συνέχεια θεωρούμε πως οι χρήστες συνεργατικά προσπαθούν να ελαχιστοποιήσουν την συνάρτηση κόστους.

Τα ακριβή βήματα του αλγορίθμου είναι τα εξής:

1. Ο Server μεταδίδει (broadcast) σε όλους τους χρήστες (ή σε ένα υποσύνολο των χρηστών) το global μοντέλο $\theta^t$.

2. Κάθε χρήστης που συμμετέχει στο συγκεκριμένο iteration χρησιμοποιεί κάποιο αλγόριθμο μάθησης (όπως το Stochastic Gradient Descent (SGD)), για να εκπαιδεύσει το μοντέλο του στα δεδομένα που έχει στην διάθεσή του, για κάποιες τοπικές επαναλήψεις.

3. Έπειτα, ο κάθε χρήστης στέλνει στον Server το μοντέλο του $\theta_i^{t+1}$.

4. Τέλος, ο Server κάνει aggregate την πληροφορία των χρηστών,

$$
\theta^{t+1} \leftarrow \sum_{i=1}^K \alpha_i \theta_i^{t+1}
$$

όπου οι συντελεστές $\alpha_i$, έχουν την ιδιότητα, $\sum_{i=1}^K \alpha_i = 1$. Συνήθως, $\alpha_i = \frac{1}{K}$.
Αφότου υπολογιστεί το νέο global model γίνεται πάλι broadcast στους χρήστες.

Τα παραπάνω βήματα επαναλαμβάνονται για ορισμένα iterations, έως ότου κάποιο κριτήριο σύγκλισης ικανοποιηθεί.

Ζητούμενα

1. (Πειραματικό Ερώτημα) Για τη συγκεκριμένη άσκηση θα χρησιμοποιήσουμε το dataset MNIST. Αποτελείται από 60.000 δείγματα στο training set και 10.000 δείγματα στο testing set. Περιέχει 10 διαφορετικές κλάσεις χειρόγραφων ψηφίων με διαστάσεις $1 \times 28 \times 28$. Αρχικά, επιλύστε το πρόβλημα της κεντρικοποιημένης μάθησης με την χρήση ενός κατάλληλου μοντέλου (π.χ. CNN, MLP). Δοκιμάστε για 2 διαφορετικές τιμές της κάθε παραμέτρου [batch size, learning rate], κρατώντας σταθερό το πλήθος των τοπικών επαναλήψεων [local epochs] και των ολικών επαναλήψεων [global iterations]. Τυπώστε την καμπύλη του training και testing loss, καθώς επίσης και την καμπύλη του testing accuracy.

2. (Πειραματικό Ερώτημα) Στην συνέχεια, στα πλαίσια της αποκεντρωμένης μάθησης, θεωρήστε πως υπάρχουν 10 χρήστες, στους οποίους θα πρέπει να μοιράσετε τα δεδομένα με IID (identically independently distributed) τρόπο. Αυτό σημαίνει πως κάθε χρήστης θα λάβει ίδιο περίπου ποσοστό δεδομένων από όλες τις διαθέσιμες κλάσεις (patterns), καθώς επίσης και ότι όλοι οι χρήστες θα διαθέτουν ίδιο πλήθος δεδομένων.
Πραγματοποιήστε πάλι τα ζητούμενα του προηγούμενου ερωτήματος.

3. (Πειραματικό Ερώτημα) Στο τέλος θα εστιάσουμε στην πιο απαιτητική πρόκληση της αποκεντρωμένης μάθησης, καθώς θα θεωρήσετε πως υπάρχουν 10 χρήστες, στους οποίους μοιράζετε τα δεδομένα με non-IID τρόπο. Αυτό σημαίνει πως κάθε χρήστης διαθέτει δεδομένα που ανήκουν αποκλειστικά σε 2 από τα 10 patterns. Πραγματοποιήστε πάλι τα ζητούμενα του πρώτου ερωτήματος.
