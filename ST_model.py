import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.init as init

class ST(nn.Module):
    def __init__(self):
        super(ST, self).__init__()

        feature_channels = 16
        width = 10
        height = 10

        self.shared_cnn2d = nn.Conv2d(in_channels=feature_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.shared_norm2d = nn.BatchNorm2d(32)
        self.shared_cnn2d1 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.shared_norm2d1 = nn.BatchNorm2d(16)

        flattened_size = 16 * width * height

        self.speed_fc1 = nn.Linear(flattened_size, 32)
        self.speed_fc2 = nn.Linear(32, 16)


        self.inflow_fc1 = nn.Linear(flattened_size, 32)
        self.inflow_fc2 = nn.Linear(32, 16)


        self.demand_fc1 = nn.Linear(flattened_size, 32)
        self.demand_fc2 = nn.Linear(32, 16)


        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)

    def normalization(self, x):
        buffer = torch.pow(x, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        normalization_constant = torch.sqrt(normp)
        output = torch.div(x, normalization_constant.view(-1, 1).expand_as(x))
        return output

    def forward(self, speed_input=None, inflow_input=None, demand_input=None):
        speed_embedding = None
        inflow_embedding = None
        demand_embedding = None


        if speed_input is not None:

            x = F.leaky_relu(self.shared_cnn2d(speed_input),0.001)
            x = self.shared_norm2d(x)
            x = F.leaky_relu(self.shared_cnn2d1(x),0.001)
            x = self.shared_norm2d1(x)

            x = x.view(x.size(0), -1)
            x = torch.tanh(self.speed_fc1(x))
            speed_embedding = F.tanh(self.speed_fc2(x))



        if inflow_input is not None:
            x = F.leaky_relu(self.shared_cnn2d(inflow_input), 0.001)
            x = self.shared_norm2d(x)
            x = F.leaky_relu(self.shared_cnn2d1(x), 0.001)
            x = self.shared_norm2d1(x)

            x = x.view(x.size(0), -1)
            x = torch.tanh(self.inflow_fc1(x))
            inflow_embedding = F.tanh(self.inflow_fc2(x))


        if demand_input is not None:
            x = F.leaky_relu(self.shared_cnn2d(demand_input), 0.001)
            x = self.shared_norm2d(x)
            x = F.leaky_relu(self.shared_cnn2d1(x), 0.001)
            x = self.shared_norm2d1(x)

            x = x.view(x.size(0), -1)
            x = torch.tanh(self.demand_fc1(x))
            demand_embedding = F.tanh(self.demand_fc2(x))

        return speed_embedding, inflow_embedding, demand_embedding



def define_model():
    model = ST()
    return model