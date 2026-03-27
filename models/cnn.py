#cnn code
import torch.nn as nn
import torch.nn.functional as F

class simpleCNN(nn.Module):
    def __init__(self): #params for class
        super(simpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1) #input channels 3 for RGB, output channels 16
        self.pool = nn.MaxPool2d(2, 2) #max pooling with kernel size 2 and stride 2

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) #input channels 16 from previous layer, output channels 32
        
        self.fc1 = nn.Linear(32 * 8 * 8, 128) #fully connected layer with input size based on the output of conv layers and output size of 128
        self.fc2 = nn.Linear(128, 10) #fully connected layer with input size of 128 and output size of 10 for the number of classes in CIFAR-10

    def forward(self, x): #function for how data moves through layers of the network
        x = self.pool(F.relu(self.conv1(x))) # 32x32 → 16x16
        x = self.pool(F.relu(self.conv2(x))) # 16x16 → 8x8
        x = x.view(-1, 32 * 8 * 8) #flatten the output from the convolutional layers
        x = F.relu(self.fc1(x)) #apply fully connected layer fc1 followed by ReLU activation
        x = self.fc2(x) #apply fully connected layer fc2 to get the final output
        return x