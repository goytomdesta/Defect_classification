import torch
import torch.nn as nn
import torch.optim as optim
from data import train_loader, test_loader, validset
from model import cl_model
import matplotlib.pyplot as plt

DEVICE = 'cuda'

PATH = "C:/Users/djmen/Desktop/m_vis/defect_detection_ResNet18.pth"
cl_model.load_state_dict(torch.load(PATH))

cl_model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = cl_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total

print(f'Test Accuracy: {100 * accuracy:.2f}%')