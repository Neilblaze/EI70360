import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.datasets as datasets
import matplotlib.pyplot as plt


# LeNet5V0 architecture
class LeNet5V0(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), stride=(1, 1)),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=(1, 1)),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5, 5), stride=(1, 1)),
            nn.Tanh()
        )

        self.linear_layer = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.flatten(1)
        x = self.linear_layer(x)
        return x


# LeNet5V1 architecture
class LeNet5V1(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), stride=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5, 5), stride=(1, 1))
        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.fc2 = nn.Linear(in_features=84, out_features=10)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.activation(x)

        x = x.flatten(1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x

# Device selection
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUM_EPOCHS = 100
BATCH_SIZE = 64
NUM_WORKERS = 2
LEARNING_RATE = 0.001

# Data transformations and loading
transform = T.Compose([
    T.ToTensor(),
    T.Resize((32, 32))
])

train_dataset = datasets.MNIST(root='./datasets', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
test_dataset = datasets.MNIST(root='./datasets', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

# Instantiate the neural network and move it to the device
net = LeNet().to(device=DEVICE)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

# Training loop
losses = list()
for epoch in range(NUM_EPOCHS):
    for data, label in train_loader:
        data = data.to(device=DEVICE)
        label = label.to(device=DEVICE)

        out = net(data)

        loss = criterion(out, label)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch: {epoch+1}-> Loss: {loss.item()}')


# Check accuracy on training and testing data
def check_accuracy(loader, model):
    num_correct = 0
    num_sample = 0
    model.eval()

    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device=DEVICE)
            y = y.to(device=DEVICE)
            score = model(x)
            prediction = score.argmax(1)
            num_correct += (prediction == y).sum()
            num_sample += score.shape[0]

        accuracy = (num_correct / num_sample) * 100
        return accuracy


print(f'Accuracy in Training: {check_accuracy(train_loader, net)}')
print(f'Accuracy in Testing: {check_accuracy(test_loader, net)}')

# Plot training losses
plt.plot(losses)
plt.show()