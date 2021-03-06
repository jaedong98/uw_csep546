import torch


class SimpleBlinkNeuralNetwork(torch.nn.Module):
    def __init__(self,
                 hiddenNodes=20,
                 conv1_input_channel=-1,
                 conv1_output_channel=5,
                 conv1_sq_convolution=12,
                 avg_pooling_kernel_size=-1,
                 avg_pooling_kernel_stride=-1,
                 conv2_input_channel=-1,
                 conv2_output_channel=5,
                 conv2_sq_convolution=12,
                 avg_pooling2_kernel_size=-1,
                 avg_pooling2_kernel_stride=2
                 ):
        super(SimpleBlinkNeuralNetwork, self).__init__()

        # convolution 1
        full_connected_one_len = 24 * 24
        self.conv1 = None
        if conv1_input_channel > 0:
            self.conv1 = torch.nn.Conv2d(conv1_input_channel,
                                         conv1_output_channel,
                                         conv1_sq_convolution)
            full_connected_one_len = conv1_output_channel * (24 - conv1_sq_convolution + 1)**2

        # avg pooling
        self.avg_pooling = None
        if avg_pooling_kernel_size > 0:
            self.avg_pooling = torch.nn.AvgPool2d(kernel_size=avg_pooling_kernel_size,
                                                  stride=avg_pooling_kernel_stride)
            if conv1_input_channel > 0:
                full_connected_one_len = conv1_output_channel * ((24 - conv1_sq_convolution + 1) // avg_pooling_kernel_size)**2
            else:
                full_connected_one_len = (24 // avg_pooling_kernel_stride)**2

        # convolution 2
        # self.conv2 = None
        # if conv2_input_channel > 0:
        #     if conv1_input_channel > 0:
        #         conv2_input_channel = conv1_output_channel
        #     self.conv2 = torch.nn.Conv2d(conv2_input_channel,
        #                                  conv2_output_channel,
        #                                  conv2_sq_convolution)
        #     if avg_pooling_kernel_size:
        #         full_connected_one_len = conv2_output_channel * (6 - conv2_sq_convolution + 1)**2
        #     else:
        #         full_connected_one_len = conv2_output_channel * (6 - conv2_sq_convolution + 1)**2

        # # convolution 2
        self.conv2 = None
        if conv2_input_channel > 0:
            if conv1_input_channel > 0:
                conv2_input_channel = conv1_output_channel
            self.conv2 = torch.nn.Conv2d(conv2_input_channel,
                                         conv2_output_channel,
                                         conv2_sq_convolution)
            dim = 24
            if conv1_sq_convolution / avg_pooling_kernel_size > 0:
                dim = conv1_sq_convolution // avg_pooling_kernel_size

            full_connected_one_len = conv2_output_channel * (dim - conv2_sq_convolution + 1) ** 2

        self.avg_pooling2 = None
        if avg_pooling2_kernel_size > 0:
            self.avg_pooling2 = torch.nn.AvgPool2d(kernel_size=avg_pooling2_kernel_size,
                                                   stride=avg_pooling2_kernel_stride)

            full_connected_one_len = conv2_output_channel * ((dim - conv2_sq_convolution + 1) // avg_pooling2_kernel_size) ** 2

        # Fully connected layer to all the down-sampled pixels to all the hidden nodes
        self.fullyConnectedOne = torch.nn.Sequential(
           #torch.nn.Linear(12*12, hiddenNodes),
           torch.nn.Linear(full_connected_one_len, hiddenNodes),
           torch.nn.Sigmoid()
           )

        # Fully connected layer from the hidden layer to a single output node
        self.outputLayer = torch.nn.Sequential(
            torch.nn.Linear(hiddenNodes, 1),
            torch.nn.Sigmoid()
            )

    def forward(self, x):
        # Apply the layers created at initialization time in order
        out = x
        if self.conv1:
            out = self.conv1(out)

        if self.avg_pooling:
            out = self.avg_pooling(out)

        if self.conv2:
            out = self.conv2(out)

        out = self.avg_pooling(out)


        out = out.reshape(out.size(0), -1)
        out = self.fullyConnectedOne(out)
        out = self.outputLayer(out)

        return out