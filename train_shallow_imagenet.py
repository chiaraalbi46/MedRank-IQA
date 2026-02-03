""" Train a shallow network on imagenet and store the trained model. """

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

# from pretraining import ShallowFeatureExtractor  # ha 1 channel in input

# --------------------------------------------------
# Small CNN model
# --------------------------------------------------

class ShallowFeatureExtractor(nn.Module):
    def __init__(self, feature_dim=256):
        super(ShallowFeatureExtractor, self).__init__()
        
        # 4 convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)   # -> (32, 224, 224)
        # self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # -> (32, 224, 224)  # one channel 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # -> (64, 112, 112)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # -> (128, 56, 56)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)# -> (256, 28, 28)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layer to produce final feature vector
        # After 4 poolings: 224 -> 112 -> 56 -> 28 -> 14
        self.fc = nn.Linear(256 * 14 * 14, feature_dim)
        # 64 --> 1 
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> (32, 112, 112)
        x = self.pool(F.relu(self.conv2(x)))  # -> (64, 56, 56)
        x = self.pool(F.relu(self.conv3(x)))  # -> (128, 28, 28)
        x = self.pool(F.relu(self.conv4(x)))  # -> (256, 14, 14)
        
        x = x.view(x.size(0), -1)             # Flatten
        features = self.fc(x)                 # Feature vector
        return features

class SmallCNN(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.features = ShallowFeatureExtractor()

        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        # x = torch.flatten(x, 1)
        return self.classifier(x)

# --------------------------------------------------
# Training function
# --------------------------------------------------

def train_one_epoch(epoch):
    model.train()
    running_loss = 0

    for images, targets in train_loader:

        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)

# --------------------------------------------------
# Validation function
# --------------------------------------------------

@torch.no_grad()
def validate():
    model.eval()
    correct = 0
    total = 0

    for images, targets in val_loader:
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)

        outputs = model(images)
        preds = outputs.argmax(dim=1)

        correct += (preds == targets).sum().item()
        total += targets.size(0)

    return correct / total

# --------------------------------------------------
# Checkpoint saving
# --------------------------------------------------

def save_checkpoint(epoch, best_acc):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_acc": best_acc
    }

    torch.save(checkpoint, os.path.join(SAVE_DIR, "last_checkpoint.pth"))

def save_pretrained():
    torch.save(model.state_dict(),
               os.path.join(SAVE_DIR, "smallcnn_imagenet_pretrained.pth"))
    

if __name__ == '__main__':


    # --------------------------------------------------
    # Configuration
    # --------------------------------------------------

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # DATA_ROOT = "/path/to/imagenet"
    DATA_ROOT = "/Prove/Bertazzini/Datasets/ImageNet"  # train e test 
    DATA_ROOT_VAL = "/Prove/Albisani/validation_imagenet"

    SAVE_DIR = "./checkpoints_shallow_imagenet"
    os.makedirs(SAVE_DIR, exist_ok=True)

    BATCH_SIZE = 256
    EPOCHS = 100
    LR = 0.01
    NUM_WORKERS = 8
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------
    # Datasets and loaders
    # --------------------------------------------------

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = datasets.ImageFolder(
        os.path.join(DATA_ROOT, "train"),
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        DATA_ROOT_VAL,  # os.path.join(DATA_ROOT, "val")
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    # --------------------------------------------------
    # Model / Loss / Optimizer
    # --------------------------------------------------

    model = SmallCNN().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=LR,
        momentum=0.9,
        weight_decay=1e-4
    )

    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------

    best_acc = 0

    for epoch in range(EPOCHS):

        start = time.time()

        train_loss = train_one_epoch(epoch)
        val_acc = validate()

        scheduler.step()

        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Acc: {val_acc:.4f}")
        print(f"Time: {time.time()-start:.1f}s")

        # Save last checkpoint
        save_checkpoint(epoch, best_acc)

        # Save best pretrained weights
        if val_acc > best_acc:
            best_acc = val_acc
            save_pretrained()
            print("Saved BEST pretrained model")

    print("Training finished.")
