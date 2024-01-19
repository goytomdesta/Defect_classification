import torch
import torch.nn as nn
import torch.optim as optim
from data import train_loader, test_loader
from model import cl_model
import matplotlib.pyplot as plt

DEVICE = 'cuda'


# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cl_model.parameters(), lr=0.001)
# Train the model and collect training history
train_loss_history = []
train_acc_history = []
# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    cl_model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = cl_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        _, predicted_train = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted_train == labels).sum().item()

    average_loss = running_loss / len(train_loader)
    accuracy_train = correct_train / total_train
    train_loss_history.append(average_loss)
    train_acc_history.append(accuracy_train)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}, Training Accuracy: {100 * accuracy_train:.2f}%')


# Save the trained model
model_path = 'defect_detection_ResNet18.pth' #Change the path and name accordingly
torch.save(cl_model.state_dict(), model_path)
print(f'Model saved to {model_path}')

# Plot training accuracy
plt.plot(train_acc_history, label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training loss
plt.plot(train_loss_history, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
