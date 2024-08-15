from __future__ import annotations
"""
This is a super simple training script to train a ResNet model AND store the gradients' norms per-component as a time-series.
It works as a CLI: you can either
1. Train the model and store the gradients
2. Process the gradients, i.e. by looking at their percentage change over time (currently in resnet_temp_viz.py)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from pathlib import Path
from datetime import datetime
# Create a results folder that is unique
results_path = Path("results") / 'gradients' / datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
results_path.mkdir(parents=True, exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transforms  TODO(Adriano) not sure if this is correct? Why are we randomly cropping size 32? Isn't CIFAR 32x32?
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

# Load pre-trained ResNet18 model
model = resnet18() # Randomly initialized: https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html

# Modify the last fully connected layer for CIFAR-10 (10 classes)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4) # TODO(Adriano) should we hyperparameter sweep this shit?

# Helper to get the norms
# NOTE: we store per-parameter norms!
def compute_grad_norm(grad):
    return torch.norm(grad, p=2).item() if grad is not None else None
def get_grad_norms(model):
    return {name: compute_grad_norm(param.grad) for name, param in model.named_parameters()}

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        grad_norms = get_grad_norms(model)
        results_file = results_path / f"gradients_norms_{epoch}_{i}.pt"
        torch.save(grad_norms, results_file)
        optimizer.step()
        
        running_loss += loss.item()
        if i % 200 == 199:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
            running_loss = 0.0

print('Finished Training')

# Save the trained model
torch.save(model.state_dict(), 'resnet18_cifar10.pth')