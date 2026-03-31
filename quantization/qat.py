import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import time
import torch    
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torch.ao.quantization import get_default_qat_qconfig, prepare_qat, convert 
import models.cnn as cnn

def evaluate_model(model, dataloader, device="cpu"):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

def measure_inference_time(model, dataloader, device="cpu", num_batches=100):
    model.eval()
    total_time = 0.0
    batch_count = 0

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            
            start_time = time.time()
            _ = model(inputs)
            end_time = time.time()

            total_time += (end_time - start_time)
            batch_count += 1

            if batch_count >= num_batches:
                break

    return total_time / batch_count     


def get_model_size_mb(model_path):
    size_bytes = Path(model_path).stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    return size_mb

def main(): 
    device = "cpu" # Quantization-aware training is typically done on CPU
    torch.backends.quantized.engine = 'onednn' # Set quantization engine to onednn for x86 CPUs

    # Data augmentation and normalization for training
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4), #data augmentation by randomly cropping the image to 32x32 with padding of 4
        transforms.RandomHorizontalFlip(), #data augmentation by randomly flipping the image horizontally
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #preprocessing input data RGB channels with mean and std deviation of 0.5
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #preprocessing input data RGB channels with mean and std deviation of 0.5
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

    #Load the pre-trained model and prepare it for quantization-aware training
    model = cnn.SimpleCNN()
    model.load_state_dict(torch.load('./models/cnnfp32.pth', map_location=device)) #load pre-trained model weights
    model.to(device)    

    model.train() #set model to training mode
    # model.fuse_model(is_qat=True) #fuse convolutional layers with batch normalization and activation layers for better quantization performance

    #QAT configuration
    model.qconfig = get_default_qat_qconfig('onednn') #set quantization configuration to use onednn backend
    model = prepare_qat(model, inplace=True) #prepare the model for quantization-aware training

    #Training loop for quantization-aware training
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) #use a smaller learning rate for fine-tuning during QAT
    criterion = nn.CrossEntropyLoss() #use cross-entropy loss for multi-class classification
    num_epochs = 5
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad() #zero the parameter gradients
            outputs = model(inputs) #forward pass
            loss = criterion(outputs, labels) #compute loss
            loss.backward() #backward pass
            optimizer.step()

            running_loss += loss.item() #accumulate loss for reporting
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(trainloader):.4f}', end='\r') #print average loss for the epoch
    torch.save(model.state_dict(), "models/cnn_qat_trained.pth") #save the quantization-aware trained model's state dictionary

    #Convert the model to a quantized version
    model.eval() #set model to evaluation mode before conversion
    quantized_model = convert(model, inplace=False) #convert the model to a quantized version

    #eval 
    accuracy = evaluate_model(quantized_model, testloader, device)
    timing = measure_inference_time(quantized_model, testloader, device)
    print(f'QAT Accuracy: {accuracy:.2f}%')
    print(f'QAT Avg Inference Time: {timing:.6f} s/batch')
    torch.save(quantized_model.state_dict(), "models/cnn_int8_qat.pth")
    qat_model_size = get_model_size_mb('models/cnn_int8_qat.pth')
    print(f'QAT Model Size: {qat_model_size:.2f} MB\n')

if __name__ == "__main__":   
    main()