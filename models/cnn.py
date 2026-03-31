#cnn code
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.quantization as quantization

class SimpleCNN(nn.Module):
    def __init__(self): #params for class
        super(SimpleCNN, self).__init__()

        self.quant = quantization.QuantStub() #quantization stub to prepare the model for quantization
        self.dequant = quantization.DeQuantStub() #dequantization stub to convert quantized output back to floating point

        self.features = nn.Sequential( #sequential container to hold convolutional layers and pooling layers
             nn.Conv2d(3, 32, kernel_size=3, padding=1), #input channels 3 for RGB, output channels 32
             nn.BatchNorm2d(32),
             nn.ReLU(), #activation function
             nn.MaxPool2d(2, 2), #max pooling with kernel size 2 and stride 2

             nn.Conv2d(32, 64, kernel_size=3, padding=1), #input channels 32 from previous layer, output channels 64
             nn.BatchNorm2d(64),
             nn.ReLU(), #activation function
             nn.MaxPool2d(2, 2), #max pooling with kernel size 2 and stride 2

            nn.Conv2d(64, 128, kernel_size=3, padding=1), #input channels 64 from previous layer, output channels 128
            nn.BatchNorm2d(128),
            nn.ReLU(), #activation function
            nn.MaxPool2d(2, 2) #max pooling with kernel size
        )

        self.classifier = nn.Sequential( #sequential container to hold fully connected layers
            nn.Flatten(), #flatten the output from the convolutional layers to feed into fully connected layers
            nn.Linear(128 * 4 * 4, 256), #fully connected layer with input size 128*4*4 and output size 256
            nn.ReLU(), #activation function
            nn.Dropout(0.3), #dropout layer with dropout probability of 0.3 to prevent overfitting
            nn.Linear(256, 10) #fully connected layer with input size 256 and output size 10 for the 10 classes in CIFAR-10
        )

    def forward(self, x): #function for how data moves through layers of the network
        x = self.quant(x) #quantize the input 
        x = self.features(x)
        x = self.classifier(x)
        x = self.dequant(x) #dequantize the output 
        return x
    
    def fuse_model(self, is_qat=False): #function to fuse layers for better quantization performance, is_qat flag indicates if the model is being prepared for quantization-aware training
        pass
        # quantization.fuse_modules(self.features, [['0', '1', '2'],
        #                                           ['4', '5', '6'],
        #                                           ['8', '9', '10']], inplace=True, is_qat=is_qat) #fuse convolutional layers with batch normalization and activation layers for better quantization performance
        # quantization.fuse_modules(self.classifier, [['1', '2']], inplace=True, is_qat=is_qat) #fuse fully connected layer with activation layer for better quantization performance