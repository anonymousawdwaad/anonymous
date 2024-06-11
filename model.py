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


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

        for param in self.parameters():
            if param.dim() >= 2:
                init.xavier_normal_(param)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1) 

        combined_conv = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.tanh(cc_i)
        f = torch.tanh(cc_f)
        o = torch.tanh(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False, predict_channel=1):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)
        self.predict_conv = nn.Conv2d(hidden_dim[-1], predict_channel, kernel_size=1)

    def forward(self, input_tensor, hidden_state=None):

        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, t, c, h, w = input_tensor.size()

        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        last_step_output = layer_output_list[-1][:, -1, :, :, :]
        prediction = torch.tanh(self.predict_conv(last_step_output))
        
        return layer_output_list, last_state_list, prediction

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param



def define_model():
    model = ST()
    return model
