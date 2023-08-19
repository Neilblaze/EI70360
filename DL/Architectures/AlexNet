import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as T

# Define the AlexNet architecture
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        # Convolutional layers
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(11, 11), stride=(4, 4)),  # Convolution
            nn.ReLU(),  # Activation
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),  # Max pooling
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), stride=(1, 1), padding=2),  # Convolution
            nn.ReLU(),  # Activation
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),  # Max pooling
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), stride=(1, 1), padding=1),  # Convolution
            nn.ReLU(),  # Activation
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), stride=(1, 1), padding=1),  # Convolution
            nn.ReLU(),  # Activation
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1),  # Convolution
            nn.ReLU(),  # Activation
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),  # Max pooling
            nn.Dropout(p=0.5)  # Dropout for regularization
        )

        # Fully connected layers
        self.linear_layer = nn.Sequential(
            nn.Linear(in_features=9216, out_features=4096),  # Fully connected
            nn.ReLU(),  # Activation
            nn.Dropout(p=0.5),  # Dropout for regularization
            nn.Linear(in_features=4096, out_features=4096),  # Fully connected
            nn.ReLU(),  # Activation
            nn.Linear(in_features=4096, out_features=1000)  # Fully connected
        )

    # Forward pass through the network
    def forward(self, x):
        x = self.conv_layer(x)
        x = x.flatten(1)
        x = self.linear_layer(x)
        return x

# Check if GPU is available, otherwise use CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters and data transformations
BATCH_SIZE = 32
NUM_WORKERS = 2
NUM_EPOCHS = 10
LEARNING_RATE = 0.01

transform = T.Compose([
    T.Resize((227, 227)),  # Resize images
    T.ToTensor()  # Convert to tensor
])

# Load training and test datasets
train_dataset = datasets.ImageNet(root='./imagenet_datasets', split='train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
test_dataset = datasets.ImageNet(root='./imagenet_datasets', split='test')
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

# Initialize the AlexNet model and move it to the appropriate device
model = AlexNet().to(device=DEVICE)

# Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(NUM_EPOCHS):
    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device=DEVICE), label.to(device=DEVICE)

        output = model(data)

        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# -----------------------------------------------------------------------------------------------



# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# import torchvision.datasets as datasets
# import torchvision.transforms as T


# class AlexNet(nn.Module):
#     def __init__(self):
#         super(AlexNet, self).__init__()

#         self.conv_layer = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(11, 11), stride=(4, 4)),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
#             nn.BatchNorm2d(96),  # Add batch normalization
#             nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), stride=(1, 1), padding=2),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
#             nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), stride=(1, 1), padding=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), stride=(1, 1), padding=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
#             nn.Dropout(p=0.5)
#         )
#         self.linear_layer = nn.Sequential(
#             nn.Linear(in_features=9216, out_features=4096),
#             nn.ReLU(),
#             nn.Dropout(p=0.5),
#             nn.Linear(in_features=4096, out_features=4096),
#             nn.ReLU(),
#             nn.Linear(in_features=4096, out_features=1000)
#         )

#     def forward(self, x):
#         x = self.conv_layer(x)
#         x = x.flatten(1)
#         x = self.linear_layer(x)
#         return x


# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# BATCH_SIZE = 32
# NUM_WORKERS = 2
# NUM_EPOCHS = 10
# LEARNING_RATE = 0.01

# transform = T.Compose([
#     T.RandomResizedCrop(227),
#     T.RandomHorizontalFlip(),
#     T.ToTensor()
# ])

# train_dataset = datasets.ImageNet(root='./imagenet_datasets', split='train', transform=transform)
# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
# test_dataset = datasets.ImageNet(root='./imagenet_datasets', split='test')
# test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

# model = AlexNet().to(device=DEVICE)

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

# for epoch in range(NUM_EPOCHS):
#     model.train()
#     for batch_idx, (data, label) in enumerate(train_loader):
#         data, label = data.to(device=DEVICE), label.to(device=DEVICE)

#         output = model(data)
#         loss = criterion(output, label)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     scheduler.step()

#     model.eval()
#     with torch.no_grad():
#         # Validation loop on the test set
#         pass

#     # Logging and model saving
#     print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Loss: {loss.item():.4f}")
