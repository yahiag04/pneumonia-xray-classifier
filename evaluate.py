import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from PneumoniaNet import PneumoniaNet, device

DATA_ROOT = "/Users/yahiaghallale/Downloads/chest_xray"
MODEL_PATH = "weights/best_pneumonia_net.pth"


test_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])


def load_model(path):
    model = PneumoniaNet().to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def evaluate_model():
    test_dir = f"{DATA_ROOT}/test"
    test_ds = datasets.ImageFolder(test_dir, transform=test_tf)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    model = load_model(MODEL_PATH)

    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device).float()

            logits = model(x).squeeze(1)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()

            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    print("\n=== CONFUSION MATRIX ===")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(all_labels, all_preds, target_names=["normal", "pneumonia"]))

    try:
        roc_auc_val = roc_auc_score(all_labels, all_probs)
        print("\n=== ROC-AUC ===")
        print(roc_auc_val)
    except Exception as e:
        print("\nROC-AUC non calcolabile:", e)

    # plot dlla confusion matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Normal", "Pneumonia"],
        yticklabels=["Normal", "Pneumonia"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("metriche_valutazione/confusion_matrix.png", dpi=300)
    plt.close()

    # plot della roc
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc_curve = auc(fpr, tpr)

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_curve:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("metriche_valutazione/roc_curve.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    evaluate_model()