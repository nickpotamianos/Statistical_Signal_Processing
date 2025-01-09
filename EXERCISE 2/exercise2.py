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
# 1. Ορισμός ενός απλού CNN για MNIST
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
# 2. Συναρτήσεις τοπικής εκπαίδευσης/αξιολόγησης
################################################################
def local_train(model, train_loader, criterion, optimizer, epochs=1, device='cpu'):
    """
    Perform local training of 'model' for 'epochs' on 'train_loader'.
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
    Evaluate 'model' on 'test_loader' and return (avg_loss, accuracy).
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
# 3. Τεμαχισμός του συνόλου δεδομένων IID μεταξύ 'num_users' χρηστών
################################################################
def split_iid(dataset, num_users=10):
    """
    Randomly splits 'dataset' into 'num_users' IID parts of equal size.
    Each user gets ~ (len(dataset)/num_users) samples.
    """
    num_items = len(dataset) // num_users
    all_indices = np.arange(len(dataset))
    np.random.shuffle(all_indices)

    subsets = []
    for i in range(num_users):
        start_idx = i * num_items
        end_idx   = start_idx + num_items
        user_indices = all_indices[start_idx:end_idx]
        subsets.append(Subset(dataset, user_indices))

    return subsets

################################################################
# 4. Συνάρτηση FedAvg
################################################################
def fed_avg(global_model, local_models, weights=None):
    """
    Performs (weighted) FedAvg of local_models into global_model in-place.
    - local_models: list of models after local training
    - weights: list of floats that sum to 1 (e.g. proportional to #samples).
    """
    global_dict = global_model.state_dict()

    # Αρχικοποίηση συσσωρευτή στο μηδέν
    for key in global_dict.keys():
        global_dict[key] = torch.zeros_like(global_dict[key])

    # Αν δεν δόθηκαν βάρη, κάνουμε απλό μέσο όρο
    if weights is None:
        weights = [1.0 / len(local_models)] * len(local_models)

    # Συσσώρευση κάθε παραμέτρου
    for w, local_model in zip(weights, local_models):
        local_dict = local_model.state_dict()
        for key in global_dict.keys():
            global_dict[key] += w * local_dict[key]

    global_model.load_state_dict(global_dict)

################################################################
# 5. Συνάρτηση για εκτέλεση ομοσπονδιακού πειράματος
################################################################
def run_federated_experiment(batch_size, learning_rate, 
                             num_users=10, global_rounds=5, 
                             local_epochs=1, 
                             device='cpu'):
    """
    Runs a Federated Learning experiment (FedAvg) on MNIST with:
      - 'num_users' IID users
      - 'batch_size', 'learning_rate'
      - 'global_rounds', 'local_epochs'
    Returns: (train_loss_history, test_loss_history, test_acc_history)
    """
    # 5.1 Φόρτωση MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset  = MNIST(root='./data', train=False, download=True, transform=transform)

    # 5.2 Τεμαχισμός του συνόλου εκπαίδευσης μεταξύ χρηστών (IID)
    subsets = split_iid(train_dataset, num_users=num_users)
    user_loaders = []
    local_num_samples = []

    for s in subsets:
        loader = DataLoader(s, batch_size=batch_size, shuffle=True)
        user_loaders.append(loader)
        local_num_samples.append(len(s))

    # Φορτωτής για το τεστ (παγκόσμιος)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # 5.3 Δημιουργία του παγκόσμιου μοντέλου
    global_model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()

    # Για αποθήκευση αποτελεσμάτων
    train_loss_history = []
    test_loss_history  = []
    test_acc_history   = []

    # 5.4 Παγκόσμιοι γύροι ομοσπονδιακής εκπαίδευσης
    for round_idx in range(global_rounds):
        # Λίστα τοπικών μοντέλων μετά την εκπαίδευση
        local_models = []

        # Τοπική εκπαίδευση κάθε χρήστη
        for user_idx in range(num_users):
            # Αντιγραφή του παγκόσμιου μοντέλου
            user_model = copy.deepcopy(global_model).to(device)
            optimizer  = optim.SGD(user_model.parameters(), lr=learning_rate, momentum=0.9)

            # Τοπική εκπαίδευση
            local_train(user_model, user_loaders[user_idx], criterion, optimizer,
                        epochs=local_epochs, device=device)
            local_models.append(user_model)

        # Βαρυτομένη FedAvg
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

        # Αποθήκευση
        train_loss_history.append(avg_train_loss)
        test_loss_history.append(test_loss)
        test_acc_history.append(test_acc)

        print(f"[Round {round_idx+1}/{global_rounds}] "
              f"batch_size={batch_size}, lr={learning_rate} | "
              f"Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}, "
              f"Test Acc: {test_acc:.2f}%")

    return train_loss_history, test_loss_history, test_acc_history

################################################################
# 6. Κύριο τμήμα - εκτελούμε αυτόματα 4 πειράματα
################################################################
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Θέλουμε 4 πειράματα:
    #   (batch_size=32, lr=0.01),
    #   (batch_size=32, lr=0.001),
    #   (batch_size=128, lr=0.01),
    #   (batch_size=128, lr=0.001).
    # Θα έχουμε για παράδειγμα global_rounds=5, local_epochs=1 για επίδειξη.
    # Μπορούμε να τα αυξήσουμε για πιο εκτεταμένη εκπαίδευση.

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
        print(f"Εκτέλεση Ομοσπονδιακού Πειράματος #{exp_number}")
        print(f"  batch_size={bs}, learning_rate={lr}, "
              f"  global_rounds={global_rounds}, local_epochs={local_epochs}")
        print("==========================================\n")

        train_losses, test_losses, test_accs = run_federated_experiment(
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

        plt.suptitle(f"Ομοσπονδιακό Πείραμα (batch_size={bs}, lr={lr})")

        # (i) Απώλεια Εκπαίδευσης
        plt.subplot(1, 3, 1)
        plt.plot(rounds, tr_losses, marker='o', label='Train Loss')
        plt.xlabel('Global Round')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Global Train Loss')

        # (ii) Απώλεια Δοκιμής
        plt.subplot(1, 3, 2)
        plt.plot(rounds, te_losses, marker='o', color='orange', label='Test Loss')
        plt.xlabel('Global Round')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Global Test Loss')

        # (iii) Ακρίβεια Δοκιμής
        plt.subplot(1, 3, 3)
        plt.plot(rounds, te_accs, marker='o', color='green', label='Test Accuracy')
        plt.xlabel('Global Round')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.title('Global Test Accuracy')

        plt.tight_layout()
        plt.show()