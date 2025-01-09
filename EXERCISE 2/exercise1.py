"""
1) Φόρτωση δεδομένων MNIST
2) Εκπαίδευση ενός μοντέλου (CNN εδώ) σε κεντροποιημένο περιβάλλον
3) Δοκιμή δύο διαφορετικών τιμών για batch_size και learning_rate
4) Εκτύπωση και σχεδίαση των καμπυλών train/test loss & test accuracy

Εκτελούμε 4 συνολικά πειράματα:
    1) batch_size=32,  learning_rate=0.01
    2) batch_size=32,  learning_rate=0.001
    3) batch_size=128, learning_rate=0.01
    4) batch_size=128, learning_rate=0.001

Μετά από κάθε πείραμα, σχεδιάζουμε:
    - Το training loss ανά εποχή
    - Το testing loss ανά εποχή
    - Το testing accuracy ανά εποχή
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# ========================
# 1. Ορισμός Μοντέλου CNN
# ========================
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
        x = self.conv_layers(x)           # [batch_size, 32, 7, 7]
        x = x.view(x.size(0), -1)         # μετατροπή σε μονοδιάστατο
        x = self.fc_layers(x)            # [batch_size, 10]
        return x

# =============================
# 2. Συνάρτηση Εκπαίδευσης
# =============================
def train_one_epoch(model, device, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, target)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    return avg_train_loss

# ===============================
# 3. Συνάρτηση Αξιολόγησης
# ===============================
def evaluate(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)

            test_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == target).sum().item()

    avg_test_loss = test_loss / len(test_loader)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    return avg_test_loss, accuracy

# =================================
# 4. Το Κύριο Πείραμα
# =================================
def run_experiment(batch_size, learning_rate, epochs=5):
    """
    Εκτελεί ένα πείραμα με τα δοσμένα batch_size και learning_rate.
    Εκπαιδεύει για 'epochs' εποχές στο σύνολο δεδομένων MNIST (κεντροποιημένο).
    Επιστρέφει:
        (train_losses, test_losses, test_accuracies) για όλες τις εποχές
    """
    # 4.1. Προετοιμασία Συνόλου Δεδομένων & DataLoader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset  = MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # 4.2. Ορισμός Μοντέλου, Συνάρτησης Απώλειας, Βελτιστοποιητή
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # 4.3. Λίστες για αποθήκευση μετρικών
    train_losses = []
    test_losses = []
    test_accuracies = []

    # 4.4. Βρόχος Εκπαίδευσης
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, device, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, device, test_loader, criterion)

        train_losses.append(train_loss)
        test_losses.append(val_loss)
        test_accuracies.append(val_acc)

        print(f"[batch_size={batch_size}, lr={learning_rate}] "
              f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, "
              f"Test Loss: {val_loss:.4f}, Test Accuracy: {val_acc:.2f}%")

    return train_losses, test_losses, test_accuracies

# ============================
# 5. Εκτέλεση και Σχεδίαση
# ============================
if __name__ == "__main__":
    """
    Θα εκτελέσουμε 4 ξεχωριστά πειράματα:
      1) (batch_size=32,  lr=0.01)
      2) (batch_size=32,  lr=0.001)
      3) (batch_size=128, lr=0.01)
      4) (batch_size=128, lr=0.001)

    Για κάθε πείραμα, αποθηκεύουμε αποτελέσματα και τα σχεδιάζουμε.
    """

    # Σύνολα υπερπαραμέτρων
    batch_sizes = [32, 128]
    learning_rates = [0.01, 0.001]
    epochs = 5  # μπορείτε να αλλάξετε σε περισσότερες εποχές (π.χ. 10 ή 15)

    experiment_results = {}
    exp_num = 1

    for bs in batch_sizes:
        for lr in learning_rates:
            print("\n===== Εκτέλεση Πειράματος #{}: batch_size={}, lr={} =====".format(exp_num, bs, lr))
            train_losses, test_losses, test_accuracies = run_experiment(bs, lr, epochs)
            experiment_results[(bs, lr)] = (train_losses, test_losses, test_accuracies)
            exp_num += 1

    # 5.1. Σχεδίαση αποτελεσμάτων (Train Loss, Test Loss, Test Acc) για κάθε πείραμα
    for (bs, lr), (tr_losses, te_losses, te_acc) in experiment_results.items():
        plt.figure(figsize=(12, 4))

        # (i) Καμπύλες Train Loss vs. Test Loss
        plt.subplot(1, 2, 1)
        plt.plot(tr_losses, label='Train Loss')
        plt.plot(te_losses, label='Test Loss')
        plt.title(f"Καμπύλες Απώλειας (batch_size={bs}, lr={lr})")
        plt.xlabel("Εποχή")
        plt.ylabel("Απώλεια")
        plt.legend()

        # (ii) Ακρίβεια στο Test
        plt.subplot(1, 2, 2)
        plt.plot(te_acc, label='Test Accuracy', color='green')
        plt.title(f"Ακρίβεια (batch_size={bs}, lr={lr})")
        plt.xlabel("Εποχή")
        plt.ylabel("Ακρίβεια (%)")
        plt.legend()

        plt.tight_layout()
        plt.show()