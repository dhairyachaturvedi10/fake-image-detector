import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

def main():
    # ── Config ──────────────────────────────────────────
    DATA_DIR   = "my_dataset"
    BATCH_SIZE = 32
    EPOCHS     = 15
    LR         = 0.001
    MODEL_OUT  = "model.pth"
    DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {DEVICE}")

    # ── Data transforms ──────────────────────────────────
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # ── Load dataset ─────────────────────────────────────
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transforms)
    print(f"Classes: {full_dataset.classes}")
    print(f"Total images: {len(full_dataset)}")

    val_size   = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    val_set.dataset.transform = val_transforms

    # num_workers=0 fixes the Windows multiprocessing error
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────
    model = models.mobilenet_v2(weights="IMAGENET1K_V1")

    for param in model.parameters():
        param.requires_grad = False

    for param in model.features[-3:].parameters():
        param.requires_grad = True

    model.classifier[1] = nn.Linear(model.last_channel, 2)
    model = model.to(DEVICE)

    # ── Loss, Optimizer, Scheduler ────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    # ── Training loop ─────────────────────────────────────
    train_accs, val_accs = [], []
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        # — Train —
        model.train()
        correct, total = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)
        train_acc = 100 * correct / total

        # — Validate —
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total   += labels.size(0)
        val_acc = 100 * correct / total

        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_OUT)
            print(f"Epoch {epoch+1}/{EPOCHS} — Train: {train_acc:.1f}%  Val: {val_acc:.1f}%  ✓ Best model saved")
        else:
            print(f"Epoch {epoch+1}/{EPOCHS} — Train: {train_acc:.1f}%  Val: {val_acc:.1f}%")

        scheduler.step()

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.1f}%")
    print(f"Model saved to {MODEL_OUT}")

    # ── Plot ──────────────────────────────────────────────
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, EPOCHS+1), train_accs, label="Train", marker='o', markersize=4)
    plt.plot(range(1, EPOCHS+1), val_accs,   label="Validation", marker='o', markersize=4)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training Progress")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("training_plot.png")
    plt.show()
    print("Plot saved to training_plot.png")

if __name__ == '__main__':
    main()