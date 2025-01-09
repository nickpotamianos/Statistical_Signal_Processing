import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import numpy as np
import copy
import matplotlib.pyplot as plt

################################################################
# 1. Ορισμός ενός απλού μοντέλου CNN για MNIST
################################################################
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(32*7*7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)  # [N, 32, 7, 7]
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)    # [N, 10]
        return x

################################################################
# 2. Τοπικές συναρτήσεις εκπαίδευσης / αξιολόγησης
################################################################
def local_train(model, train_loader, criterion, optimizer, epochs=1, device='cpu'):
    """
    Εκτελεί τοπική εκπαίδευση του 'model' για 'epochs' χρησιμοποιώντας το 'train_loader'.
    """
    model.train()
    for _ in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

def evaluate(model, test_loader, criterion, device='cpu'):
    """
    Αξιολογεί το 'model' στο 'test_loader' και επιστρέφει (avg_loss, accuracy).
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()

            _, predicted = torch.max(output, dim=1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

    avg_test_loss = test_loss / len(test_loader)
    accuracy = 100.0 * correct / total
    return avg_test_loss, accuracy

################################################################
# 3. Μη-IID κατανομή: κάθε χρήστης παίρνει ακριβώς 2 κλάσεις
################################################################
def split_non_iid(dataset, num_users=10):
    """
    Υποθέτουμε 10 κλάσεις στο MNIST (ψηφία 0..9) και 10 χρήστες.
    Κάθε χρήστης παίρνει ακριβώς 2 κλάσεις, με ζευγάρια που επαναλαμβάνονται για να καλύψουμε τους 10 χρήστες.
    pairs_of_classes = [[0,1], [0,1], [2,3], [2,3], [4,5], [4,5], [6,7], [6,7], [8,9], [8,9]]
    
    Για κάθε ζευγάρι, μαζεύουμε όλα τα δείγματα από αυτές τις 2 κλάσεις από το 'dataset'.
    Επιστρέφει λίστα από 10 Subsets, ένα για κάθε χρήστη.
    """
    pairs_of_classes = [
        [0,1], [0,1],
        [2,3], [2,3],
        [4,5], [4,5],
        [6,7], [6,7],
        [8,9], [8,9]
    ]
    subsets = []

    targets = np.array(dataset.targets)
    for pair in pairs_of_classes:
        c1, c2 = pair
        idx = np.where((targets == c1) | (targets == c2))[0]
        subset = Subset(dataset, idx)
        subsets.append(subset)

    return subsets

################################################################
# 4. Συνάρτηση FedAvg
################################################################
def fed_avg(global_model, local_models, weights=None):
    """
    Εκτελεί (weighted) FedAvg των local_models στο global_model επί τόπου.
    - local_models: λίστα από μοντέλα μετά από τοπική εκπαίδευση
    - weights: λίστα με αριθμούς που αθροίζουν στο 1 (π.χ. ανάλογοι του αριθμού δειγμάτων).
    """
    global_dict = global_model.state_dict()

    # Αρχικοποίηση συσσωρευτή στο μηδέν
    for key in global_dict.keys():
        global_dict[key] = torch.zeros_like(global_dict[key])

    # Αν δεν δοθούν βάρη, κάνουμε απλό μέσο όρο
    if weights is None:
        weights = [1.0 / len(local_models)] * len(local_models)

    # Συσσώρευση κάθε παραμέτρου
    for w, local_model in zip(weights, local_models):
        local_dict = local_model.state_dict()
        for key in global_dict.keys():
            global_dict[key] += w * local_dict[key]

    global_model.load_state_dict(global_dict)

################################################################
# 5. Ομοσπονδιακό πείραμα (μη-IID, κάθε χρήστης έχει 2 κλάσεις)
################################################################
def run_federated_experiment_non_iid(batch_size, learning_rate, 
                                     num_users=10, global_rounds=5, 
                                     local_epochs=1, 
                                     device='cpu'):
    """
    Εκτελεί ένα πείραμα Ομοσπονδιακής Μάθησης στο MNIST με μη-IID δεδομένα:
      - 'num_users' = 10, κάθε χρήστης έχει δεδομένα από ακριβώς 2 κλάσεις.
      - 'batch_size', 'learning_rate'
      - 'global_rounds', 'local_epochs'
    Επιστρέφει: (train_loss_history, test_loss_history, test_acc_history)
    """
    # 5.1 Φόρτωση MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset  = MNIST(root='./data', train=False, download=True, transform=transform)

    # 5.2 Διαμοιρασμός του train set σε χρήστες (μη-IID, 2 κλάσεις ανά χρήστη)
    subsets = split_non_iid(train_dataset, num_users=num_users)
    user_loaders = []
    local_num_samples = []

    for s in subsets:
        loader = DataLoader(s, batch_size=batch_size, shuffle=True)
        user_loaders.append(loader)
        local_num_samples.append(len(s))

    # Φορτωτής δοκιμής (παγκόσμιος)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # 5.3 Δημιουργία του παγκόσμιου μοντέλου
    global_model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()

    # Για αποθήκευση των αποτελεσμάτων
    train_loss_history = []
    test_loss_history  = []
    test_acc_history   = []

    # 5.4 Γύροι Ομοσπονδιακής Εκπαίδευσης
    for round_idx in range(global_rounds):
        local_models = []

        # Τοπική εκπαίδευση για κάθε χρήστη
        for user_idx in range(num_users):
            user_model = copy.deepcopy(global_model).to(device)
            optimizer  = optim.SGD(user_model.parameters(), lr=learning_rate, momentum=0.9)

            local_train(user_model, user_loaders[user_idx], criterion, optimizer,
                        epochs=local_epochs, device=device)
            local_models.append(user_model)

        # Weighted FedAvg
        total_samples = sum(local_num_samples)
        weights = [s / total_samples for s in local_num_samples]
        fed_avg(global_model, local_models, weights=weights)

        # Υπολογισμός train loss του παγκόσμιου μοντέλου (σε όλα τα τοπικά δεδομένα)
        global_model.eval()
        running_loss = 0.0
        total_data   = 0
        with torch.no_grad():
            for user_idx in range(num_users):
                for data, target in user_loaders[user_idx]:
                    data, target = data.to(device), target.to(device)
                    output = global_model(data)
                    loss   = criterion(output, target)
                    bsz    = data.size(0)
                    running_loss += loss.item() * bsz
                    total_data   += bsz
        avg_train_loss = running_loss / total_data if total_data>0 else 0.0

        # Αξιολόγηση στο σύνολο τεστ
        test_loss, test_acc = evaluate(global_model, test_loader, criterion, device=device)

        # Καταγραφή
        train_loss_history.append(avg_train_loss)
        test_loss_history.append(test_loss)
        test_acc_history.append(test_acc)

        print(f"[Round {round_idx+1}/{global_rounds}] "
              f"Μη-IID (2 κλάσεις/χρήστη), batch_size={batch_size}, lr={learning_rate} | "
              f"Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}, "
              f"Test Acc: {test_acc:.2f}%")

    return train_loss_history, test_loss_history, test_acc_history

################################################################
# 6. Κύριο τμήμα - εκτελούμε αυτόματα 4 πειράματα (όπως Q1)
################################################################
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Εκτελούμε τις ίδιες 4 συνδυαστικές υπερπαραμέτρων όπως πριν:
    #   (batch_size=32, lr=0.01),
    #   (batch_size=32, lr=0.001),
    #   (batch_size=128, lr=0.01),
    #   (batch_size=128, lr=0.001).
    # Θα ορίσουμε π.χ. global_rounds=5, local_epochs=1 για επίδειξη.
    # Μπορείτε να τα αυξήσετε για πιο εκτεταμένη εκπαίδευση.

    exp_params = [
        (32,   0.01),
        (32,   0.001),
        (128,  0.01),
        (128,  0.001),
    ]

    global_rounds = 5
    local_epochs  = 1
    num_users     = 10

    results_dict = {}
    exp_number = 1

    for (bs, lr) in exp_params:
        print("\n==========================================")
        print(f"Εκτέλεση Ομοσπονδιακού Μη-IID Πειράματος #{exp_number}")
        print(f"  batch_size={bs}, learning_rate={lr}, "
              f"  global_rounds={global_rounds}, local_epochs={local_epochs}, num_users={num_users}")
        print("==========================================\n")

        train_losses, test_losses, test_accs = run_federated_experiment_non_iid(
            batch_size=bs,
            learning_rate=lr,
            num_users=num_users,
            global_rounds=global_rounds,
            local_epochs=local_epochs,
            device=device
        )

        results_dict[(bs, lr)] = (train_losses, test_losses, test_accs)
        exp_number += 1

    # -----------------------------------------------------------
    # 7. Σχεδίαση καμπυλών για κάθε πείραμα
    # -----------------------------------------------------------
    rounds = np.arange(1, global_rounds+1)

    for (bs, lr), (tr_losses, te_losses, te_accs) in results_dict.items():
        plt.figure(figsize=(14, 4))
        plt.suptitle(f"Ομοσπονδιακό Μη-IID (Χρήστης Έχει 2 Κλάσεις) - (batch_size={bs}, lr={lr})")

        # (i) Απώλεια Εκπαίδευσης
        plt.subplot(1, 3, 1)
        plt.plot(rounds, tr_losses, marker='o', label='Train Loss')
        plt.xlabel('Global Round')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Global Train Loss')

        # (ii) Test Loss
        plt.subplot(1, 3, 2)
        plt.plot(rounds, te_losses, marker='o', color='orange', label='Test Loss')
        plt.xlabel('Global Round')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Global Test Loss')

        # (iii) Test Accuracy
        plt.subplot(1, 3, 3)
        plt.plot(rounds, te_accs, marker='o', color='green', label='Test Accuracy')
        plt.xlabel('Global Round')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.title('Global Test Accuracy')

        plt.tight_layout()
        plt.show()