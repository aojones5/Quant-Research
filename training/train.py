#pyTorch Loads in CIFAR-10 dataset
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import models.cnn as cnn
import torchvision.transforms as transforms

# Convert the images to tensors and normalize them
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #preprocessing input data RGB channels with mean and std deviation of 0.5
])

# Load dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

#Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #check if GPU is available and set device accordingly
model = cnn.simpleCNN().to(device) #initialize the CNN model and move it to the appropriate device
criterion = nn.CrossEntropyLoss() #use cross-entropy loss for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001) 

#Training loop
num_epochs = 5
for epoch in range(num_epochs):
    running_loss = 0.0

    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device) #move inputs and labels to the appropriate device

        optimizer.zero_grad() #zero the parameter gradients

        outputs = model(inputs) #forward pass
        loss = criterion(outputs, labels) #compute loss
        loss.backward() #backward pass
        optimizer.step() #update weights

        running_loss += loss.item() #accumulate loss for reporting

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(trainloader):.4f}') #print average loss for the epoch

#Evaluate the model on the test set
correct = 0
total = 0

with torch.no_grad(): #disable gradient calculation for evaluation
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device) #move inputs and labels to the appropriate device
        outputs = model(inputs) #forward pass
        _, predicted = torch.max(outputs.data, 1) #get predicted class with highest score
        total += labels.size(0) #update total count of samples
        correct += (predicted == labels).sum().item() #update count of correct predictions

print(f'Accuracy: {100 * correct / total:.2f}%') #print accuracy on the test set

torch.save(model.state_dict(), 'models/cnnfp32.pth') #save the trained model's state dictionary