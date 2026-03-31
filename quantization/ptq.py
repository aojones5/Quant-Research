import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import copy
import time
import torch  
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.ao.quantization import get_default_qconfig, prepare, convert
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
    device = "cpu" # Quantization is typically done on CPU
    torch.backends.quantized.engine = 'onednn' # Set quantization engine to onednn for x86 CPUs

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

    #load trained FP32 model
    fp32_model = cnn.SimpleCNN()
    fp32_model.load_state_dict(torch.load("models/cnnfp32.pth", map_location=device))
    fp32_model.eval()
    fp32_model.fuse_model() #fuse layers for quantization

    #save FP32 model size and accuracy
    torch.save(fp32_model.state_dict(), 'models/cnnfp32quant.pth')
    fp32_model_size = get_model_size_mb('models/cnnfp32quant.pth')

    fp32_model_accuracy = evaluate_model(fp32_model, testloader, device)
    fp32_inference_time = measure_inference_time(fp32_model, testloader, device)

    #prepare PTQ model
    ptq_model = copy.deepcopy(fp32_model)
    ptq_model.eval()
    ptq_model.qconfig = get_default_qconfig('onednn') #set quantization configuration for onednn backend    

    prepared_model = prepare(ptq_model)

    #calibrate with training data
    with torch.no_grad():
        for i, (inputs, _) in enumerate(trainloader):
            prepared_model(inputs)
            if i >= 99: #calibrate with 100 batches
                break

    quantized_model = convert(prepared_model)

    #save quantized model size and accuracy
    torch.save(quantized_model.state_dict(), 'models/cnn_int8_ptq.pth')
    ptq_model_size = get_model_size_mb('models/cnn_int8_ptq.pth')

    ptq_model_accuracy = evaluate_model(quantized_model, testloader, device)
    ptq_inference_time = measure_inference_time(quantized_model, testloader, device)

    print("\n--- PTQ Results ---")
    print(f"FP32 Accuracy: {fp32_model_accuracy:.2f}%")
    print(f"PTQ  Accuracy: {ptq_model_accuracy:.2f}%")
    print(f"FP32 Size:     {fp32_model_size:.2f} MB")
    print(f"PTQ  Size:     {ptq_model_size:.2f} MB")
    print(f"FP32 Avg Inference Time: {fp32_inference_time:.6f} s/batch")
    print(f"PTQ  Avg Inference Time: {ptq_inference_time:.6f} s/batch")


if __name__ == "__main__":
    main()  