import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# PATH DEL DATASET
DATA_ROOT = "/Users/yahiaghallale/Downloads/chest_xray"

# DEVICE
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using device:", device)


# MODELLO CNN
class PneumoniaNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)

        self.conv2a = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2a   = nn.BatchNorm2d(32)
        self.conv2b = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2b   = nn.BatchNorm2d(32)

        self.conv3a = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3a   = nn.BatchNorm2d(64)
        self.conv3b = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3b   = nn.BatchNorm2d(64)

        self.conv4a = nn.Conv2d(64, 128, 3, padding=1)
        self.bn4a   = nn.BatchNorm2d(128)
        self.conv4b = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4b   = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(128, 64)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = self.pool(x)

        x = F.relu(self.bn3a(self.conv3a(x)))
        x = F.relu(self.bn3b(self.conv3b(x)))
        x = self.pool(x)

        x = F.relu(self.bn4a(self.conv4a(x)))
        x = F.relu(self.bn4b(self.conv4b(x)))
        x = self.pool(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x


# TRASFORMAZIONI
train_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(7),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

test_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])


# LOADING DEL MODEL
def load_trained_model(model_path):
    model = PneumoniaNet().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


# FASE DI TRAINING
def train_model():

    train_ds = datasets.ImageFolder(os.path.join(DATA_ROOT, "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(DATA_ROOT, "val"),   transform=test_tf)
    test_ds  = datasets.ImageFolder(os.path.join(DATA_ROOT, "test"),  transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False)

    model = PneumoniaNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    best_val_loss = float("inf")
    patience = 5
    no_improve = 0
    best_path = "weights/best_pneumonia_net1.pth"

    print("Classi:", train_ds.classes)

    for epoch in range(30):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.float().to(device)

            optimizer.zero_grad()
            out = model(x).squeeze(1)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = (torch.sigmoid(out) >= 0.5).long()
            correct += (preds == y.long()).sum().item()
            total += x.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        # VALIDAZIONE
        model.eval()
        val_loss = 0
        val_corr = 0
        val_tot = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.float().to(device)

                out = model(x).squeeze(1)
                loss = criterion(out, y)

                val_loss += loss.item() * x.size(0)
                preds = (torch.sigmoid(out) >= 0.5).long()

                val_corr += (preds == y.long()).sum().item()
                val_tot += x.size(0)

        val_loss /= val_tot
        val_acc = val_corr / val_tot

        print(f"Epoch {epoch+1}/30 | Train {train_loss:.4f} acc {train_acc:.4f} | "
              f"Val {val_loss:.4f} acc {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), best_path)
            print("modello migliore trovato")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stop")
                break

    print("\n=== TESTING MODEL ===")
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.float().to(device)

            logits = model(x).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

    print("Confusion matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print(classification_report(all_labels, all_preds, target_names=["normal", "pneumonia"]))
    print("ROC-AUC:", roc_auc_score(all_labels, all_probs))


# PREDICT IMAGE (UNA SOLA)
def predict_single(img_path, model):
    img = Image.open(img_path).convert("L")
    x = test_tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(x).squeeze()
        prob = torch.sigmoid(logit).item()

    label = "pneumonia" if prob >= 0.5 else "normal"

    print(f"Immagine: {img_path}")
    print(f"Predizione: {label} | P(pneumonia) = {prob:.4f}")

    return label, prob, x, img



if __name__ == "__main__":
    train_model()




