import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
from PIL import Image

# 1. Custom Dataset Class
class HAM10000(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.df.iloc[index]['path']
        image = Image.open(img_path).convert('RGB')
        label = self.df.iloc[index]['label_idx']
        if self.transform:
            image = self.transform(image)
        return image, label

# 2. Setup Transforms (Augmentation)
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. Load Model (Transfer Learning with MobileNetV2)
model = models.mobilenet_v2(pretrained=True)

# Freeze layers
for param in model.parameters():
    param.requires_grad = False

# Replace the final classifier (7 classes)
model.classifier[1] = nn.Sequential(
    nn.Linear(model.last_channel, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 7)
)

# 4. Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier[1].parameters(), lr=0.001)

# To save:
# torch.save(model.state_dict(), 'backend/models/skin_model.pth')