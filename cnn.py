import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, dim_in, hidden_dim, kernel_size, stride, num_layers=1):
        super(CNN, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_layers = num_layers
        
        layers = []
        current_dim_in = dim_in
        
        for i in range(num_layers):
            if i == 0:
                in_channels = dim_in
            else:
                in_channels = hidden_dim
            
            layers.append(nn.Conv2d(in_channels, hidden_dim, 
                                  kernel_size=kernel_size, stride=stride))
            
            layers.append(nn.BatchNorm2d(hidden_dim))
            
            layers.append(nn.ReLU(inplace=True))
            
            layers.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride))
        
        self.cnn = nn.Sequential(*layers)
    
    def forward(self, PI):
        feature = self.cnn(PI)
        return feature

    def cnn_output_dim(self, PI_dim):
        current_dim = PI_dim
        
        for _ in range(self.num_layers):
            conv_dim = int((current_dim - self.kernel_size) / self.stride) + 1
            current_dim = int((conv_dim - self.kernel_size) / self.stride) + 1
            
        return current_dim