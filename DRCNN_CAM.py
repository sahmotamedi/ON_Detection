import torch
import torch.nn as nn


# noinspection PyTypeChecker
class DRCNNPaper(nn.Module):
    """
    This is the dilated residual network used in the paper (using the default parameters of the __init__ method) for the classification of healthy controls and optic neuritis eyes
    """
    def __init__(self, input_size=(2,512,512), num_channels=(32,32,64,64,128,128,128,128,128,128,128), kernel_size=3, cnn_bias=True, dilation=(1,2,1,2,1,2,4,8,8,16,32), fc_bias=False, dropout_rate=0.4):
        super().__init__()

        # Define network components
        self.conv_1 = nn.Conv2d(input_size[0], num_channels[0], kernel_size=kernel_size, stride=1, padding='same', bias=cnn_bias, dilation=dilation[0])
        self.conv_2 = nn.Conv2d(num_channels[0], num_channels[1], kernel_size=kernel_size, stride=1, padding='same', bias=cnn_bias, dilation=dilation[1])
        self.conv_3 = nn.Conv2d(num_channels[0] + num_channels[1], num_channels[2], kernel_size=kernel_size, stride=1, padding='same', bias=cnn_bias, dilation=dilation[2])
        self.conv_4 = nn.Conv2d(num_channels[2], num_channels[3], kernel_size=kernel_size, stride=1, padding='same', bias=cnn_bias, dilation=dilation[3])
        self.conv_5 = nn.Conv2d(num_channels[2]+num_channels[3], num_channels[4], kernel_size=kernel_size, stride=1, padding='same', bias=cnn_bias, dilation=dilation[4])
        self.conv_6 = nn.Conv2d(num_channels[4], num_channels[5], kernel_size=kernel_size, stride=1, padding='same', bias=cnn_bias, dilation=dilation[5])
        self.conv_7 = nn.Conv2d(num_channels[4]+num_channels[5], num_channels[6], kernel_size=kernel_size, stride=1, padding='same', bias=cnn_bias, dilation=dilation[6])
        self.conv_8 = nn.Conv2d(num_channels[6], num_channels[7], kernel_size=kernel_size, stride=1, padding='same', bias=cnn_bias, dilation=dilation[7])
        self.conv_9 = nn.Conv2d(num_channels[6]+num_channels[7], num_channels[8], kernel_size=kernel_size, stride=1, padding='same', bias=cnn_bias, dilation=dilation[8])
        self.conv_10 = nn.Conv2d(num_channels[8], num_channels[9], kernel_size=kernel_size, stride=1, padding='same', bias=cnn_bias, dilation=dilation[9])
        self.conv_11 = nn.Conv2d(num_channels[8]+num_channels[9], num_channels[10], kernel_size=kernel_size, stride=1, padding='same', bias=cnn_bias, dilation=dilation[10])
        self.fc = nn.Conv2d(num_channels[10], 1, kernel_size=(int(input_size[1]/64), int(input_size[2]/64)), bias=fc_bias)
        self.dropout = nn.Dropout(dropout_rate)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.activation_cnn = nn.ReLU()
        # The sigmoid function was omitted from here as required by grad CAM
        self.gradients = None

    def activations_hook(self, grad):
        # This function saves the gradient of the output with regard to the last activation layer
        self.gradients = grad

    def forward(self, x):
        # Perform the forward pass
        out = self.activation_cnn(self.conv_1(x))
        out_temp = self.activation_cnn(self.conv_2(out))
        out = self.dropout(self.max_pool(torch.concat((out, out_temp), dim=1)))
        out = self.activation_cnn(self.conv_3(out))
        out_temp = self.activation_cnn(self.conv_4(out))
        out = self.dropout(self.max_pool(torch.concat((out, out_temp), dim=1)))
        out = self.activation_cnn(self.conv_5(out))
        out_temp = self.activation_cnn(self.conv_6(out))
        out = self.dropout(self.max_pool(torch.concat((out, out_temp), dim=1)))
        out = self.activation_cnn(self.conv_7(out))
        out_temp = self.activation_cnn(self.conv_8(out))
        out = self.dropout(self.max_pool(torch.concat((out, out_temp), dim=1)))
        out = self.activation_cnn(self.conv_9(out))
        out_temp = self.activation_cnn(self.conv_10(out))
        out = self.dropout(self.max_pool(torch.concat((out, out_temp), dim=1)))
        out = self.activation_cnn(self.conv_11(out))
        h = out.register_hook(self.activations_hook)
        out = self.dropout(self.max_pool(out))
        out = self.fc(out)

        return out

    def get_activation_gradients(self):
        return self.gradients

    def get_activations(self, x):
        # This method returns the feature maps of the last convolution after the activation
        out = self.activation_cnn(self.conv_1(x))
        out_temp = self.activation_cnn(self.conv_2(out))
        out = self.dropout(self.max_pool(torch.concat((out, out_temp), dim=1)))
        out = self.activation_cnn(self.conv_3(out))
        out_temp = self.activation_cnn(self.conv_4(out))
        out = self.dropout(self.max_pool(torch.concat((out, out_temp), dim=1)))
        out = self.activation_cnn(self.conv_5(out))
        out_temp = self.activation_cnn(self.conv_6(out))
        out = self.dropout(self.max_pool(torch.concat((out, out_temp), dim=1)))
        out = self.activation_cnn(self.conv_7(out))
        out_temp = self.activation_cnn(self.conv_8(out))
        out = self.dropout(self.max_pool(torch.concat((out, out_temp), dim=1)))
        out = self.activation_cnn(self.conv_9(out))
        out_temp = self.activation_cnn(self.conv_10(out))
        out = self.dropout(self.max_pool(torch.concat((out, out_temp), dim=1)))
        out = self.activation_cnn(self.conv_11(out))

        return out
