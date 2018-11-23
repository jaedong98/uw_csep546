from Assignment4Support import draw_accuracies
from torch import nn
import torch.nn.functional as F


class LeNet(nn.Module):

    def __init__(self, hiddenNodes=20):
        super(LeNet, self).__init__()
        # input channel = 1, output channel = 6, kernel_size = 5
        # input size = (32, 32),
        self.conv1 = nn.Conv2d(1, 12, 5)
        # output size = (28, 28)  # (28 = 32 - 5 + 1)

        # pulling (2, 2)
        # output size = (14, 14)

        # input channel = 6, output channel = 16, kernel_size = 5
        # input size = (14, 14),
        self.conv2 = nn.Conv2d(12, 16, 5)
        # output size = (10, 10)  # (10 = 14 - 5 + 1)

        # pulling (2, 2)
        # output size = (5, 5)

        # input dim = 16*5*5, output dim = hiddenNodes
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
            nn.Linear(144, hiddenNodes),
            nn.Sigmoid()
        )
        # input dim = 84, output dim = 10
        self.fc3 = nn.Sequential(
            nn.Linear(hiddenNodes, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
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

    torch.manual_seed(1)
    model = LeNet()
    lossFunction = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

    # Load the images and then convert them into tensors (no normalization)
    xTrainImages = [Image.open(path) for path in xTrainRaw]
    xTrain = torch.stack([transforms.ToTensor()(image) for image in xTrainImages])

    xTestImages = [Image.open(path) for path in xTestRaw]
    xTest = torch.stack([transforms.ToTensor()(image) for image in xTestImages])

    yTrain = torch.Tensor([[yValue] for yValue in yTrainRaw])
    yTest = torch.Tensor([[yValue] for yValue in yTestRaw])

    configuration = 'conv1_12out_left_only_w_rot_500'
    report_fname = os.path.join(report_path, '{}.md'.format(configuration))
    loss_fname = os.path.join(report_path, 'loss_{}.png'.format(configuration))
    accu_fname = os.path.join(report_path, 'accuracy_{}.png'.format(configuration))

    ITERATION = 500
    losses = []
    accuracies = []
    with open(report_fname, 'w') as reporter:
        for i in range(ITERATION):
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
    simpleAccuracy = ev.accuracy

    #accuracies.append((ITERATION, ev.accuracy))
    #accuracies.append((400, 100))

    #draw_accuracies(accuracies, 'Iterations', 'Accuracy', configuration, accu_fname, [])
