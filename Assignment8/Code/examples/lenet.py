from Assignment4Support import draw_accuracies
from torch import nn
import torch.nn.functional as F


class LeNet(nn.Module):

    def __init__(self,
                 conv1_output_channel=6,
                 conv2_output_channel=16,
                 conv_kernel_size=5,
                 pooling_size=2,
                 hiddenNodes=20):

        self.config_name = 'c1oc{}_c2oc{}_cksize{}_psize{}_hnodes{}'\
                            .format(conv1_output_channel,
                                    conv2_output_channel,
                                    conv_kernel_size,
                                    pooling_size,
                                    hiddenNodes)

        super(LeNet, self).__init__()
        # input channel = 1, output channel = 6, kernel_size = 5
        # input size = (24, 24),
        self.conv1 = nn.Conv2d(1, conv1_output_channel, conv_kernel_size)
        conv1_out_dim = 24 - conv_kernel_size + 1
        # output size = (20, 20)  # (20 = 24 - 5 + 1)

        # pooling (2, 2)
        # output size = (10, 10)
        pooling1_out_dim = conv1_out_dim // pooling_size

        # input channel = 6, output channel = 16, kernel_size = 5
        # input size = (10, 10),
        self.conv2 = nn.Conv2d(conv1_output_channel, conv2_output_channel, conv_kernel_size)
        conv2_out_dim = pooling1_out_dim - conv_kernel_size + 1
        # output size = (6, 6)  # (6 = 10 - 5 + 1)

        # pulling (2, 2)
        # output size = (3, 3)
        pooling2_out_dim = conv2_out_dim // pooling_size

        # input dim = 16*3*3, output dim = hiddenNodes
        # self.fc1 = nn.Sequential(
        #     nn.Linear(144, 120),
        #     nn.Sigmoid()
        # )
        # # input dim = 120, output dim = 84
        # self.fc2 = nn.Sequential(
        #     nn.Linear(120, hiddenNodes),
        #     nn.Sigmoid()
        # )

        self.fc1 = nn.Sequential(
            nn.Linear(conv2_output_channel * pooling2_out_dim**2, hiddenNodes),
            nn.Sigmoid()
        )
        # input dim = 84, output dim = 10
        self.fc3 = nn.Sequential(
            nn.Linear(hiddenNodes, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        dropout = nn.Dropout2d(p=0.2)
        x = F.max_pool2d(self.conv1(x), (2, 2), stride=2)
        #x = nn.Softmax2d()(x)
        x = F.max_pool2d(self.conv2(x), (2, 2), stride=2)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        # x = self.fc2(x)
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    from Assignment8.Code import kDataPath, report_path
    import Assignment5Support

    (xRaw, yRaw) = Assignment5Support.LoadRawData(kDataPath,
                                                  includeLeftEye=True,
                                                  includeRightEye=False,
                                                  augments=['rot'])
    #xRaw = xRaw[: len(xRaw) // 2]
    #yRaw = yRaw[: len(yRaw) // 2]
    (xTrainRaw, yTrainRaw, xTestRaw, yTestRaw) = Assignment5Support.TrainTestSplit(xRaw, yRaw, percentTest=.25)

    print("Train is %f percent closed." % (sum(yTrainRaw) / len(yTrainRaw)))
    print("Test is %f percent closed." % (sum(yTestRaw) / len(yTestRaw)))

    import os
    from PIL import Image
    import torchvision.transforms as transforms
    import torch
    from EvaluationsStub import Evaluation


    # Load the images and then convert them into tensors (no normalization)
    xTrainImages = [Image.open(path) for path in xTrainRaw]
    xTrain = torch.stack([transforms.ToTensor()(image) for image in xTrainImages])

    xTestImages = [Image.open(path) for path in xTestRaw]
    xTest = torch.stack([transforms.ToTensor()(image) for image in xTestImages])

    yTrain = torch.Tensor([[yValue] for yValue in yTrainRaw])
    yTest = torch.Tensor([[yValue] for yValue in yTestRaw])

    output_path = os.path.join(report_path, 'param_sweeps')
    highest_accuracy = -1
    highest_accuracy_fname = ''

    for iteration in [1000]:
        for conv1_output_channel in [6, 8, 10]:
            for conv2_output_channel in [16, 20, 24, 28]:
                for hiddenNodes in [40]:
                    for conv_kernel_size in [3, 4, 5]:

                        torch.manual_seed(1)
                        model = LeNet(conv1_output_channel=conv1_output_channel,
                                      conv2_output_channel=conv2_output_channel,
                                      hiddenNodes=hiddenNodes,
                                      conv_kernel_size=conv_kernel_size,
                                      pooling_size=2)
                        lossFunction = torch.nn.MSELoss(reduction='sum')
                        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

                        configuration = '{}_rot_iter{}'.format(model.config_name, iteration)
                        report_fname = os.path.join(output_path, '{}.md'.format(configuration))
                        loss_fname = os.path.join(output_path, 'loss_{}.png'.format(configuration))
                        accu_fname = os.path.join(output_path, 'accuracy_{}.png'.format(configuration))

                        losses = []
                        accuracies = []
                        with open(report_fname, 'w') as reporter:
                            for i in range(iteration):
                                # Do the forward pass
                                yTrainPredicted = model(xTrain)

                                # Compute the training set loss
                                loss = lossFunction(yTrainPredicted, yTrain)
                                print(i, loss.item())
                                losses.append((i, loss.item()))

                                # Reset the gradients in the network to zero
                                optimizer.zero_grad()

                                # Backprop the errors from the loss on this iteration
                                loss.backward()

                                # Do a weight update step
                                optimizer.step()

                                if i > 0 and i % 100 == 0:
                                    yTestPredicted = model(xTest)
                                    yPred = [1 if pred > 0.5 else 0 for pred in yTestPredicted]
                                    ev = Evaluation(yTest, yPred)
                                    print("Accuracy simple:", ev.accuracy)
                                    accuracies.append((i, ev.accuracy))
                                    reporter.write(str(ev))
                                    reporter.write('\n')

                            yTestPredicted = model(xTest)

                            yPred = [1 if pred > 0.5 else 0 for pred in yTestPredicted]

                            ev = Evaluation(yTest, yPred)
                            # print("Accuracy simple:", Evaluations.Accuracy(yTest, yPred))
                            # simpleAccuracy = Evaluations.Accuracy(yTest, yPred)
                            print("Accuracy simple:", ev.accuracy)
                            reporter.write(str(ev))
                            if ev.accuracy > highest_accuracy:
                                highest_accuracy = ev.accuracy
                                highest_accuracy_fname = configuration
                        accuracies.append((iteration, ev.accuracy))
                        draw_accuracies([accuracies], 'Iterations', 'Accuracy', configuration, accu_fname, [])

                        draw_accuracies([losses], 'Iterations', 'Losses', configuration, loss_fname, [], data_pt='-')

    print("Highest Accuracy: {}".format(highest_accuracy))
    print(highest_accuracy_fname)
