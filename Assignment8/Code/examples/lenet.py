from torch import nn
import torch.nn.functional as F


class LeNet(nn.Module):

    def __init__(self, hiddenNodes=20):
        super(LeNet, self).__init__()
        # input channel = 1, output channel = 6, kernel_size = 5
        # input size = (32, 32),
        self.conv1 = nn.Conv2d(1, 6, 5)
        # output size = (28, 28)  # (28 = 32 - 5 + 1)

        # pulling (2, 2)
        # output size = (14, 14)

        # input channel = 6, output channel = 16, kernel_size = 5
        # input size = (14, 14),
        self.conv2 = nn.Conv2d(6, 16, 5)
        # output size = (10, 10)  # (10 = 14 - 5 + 1)

        # pulling (2, 2)
        # output size = (5, 5)

        # input dim = 16*5*5, output dim = hiddenNodes
        self.fc1 = nn.Linear(144, 120)
        # input dim = 120, output dim = 84
        self.fc2 = nn.Linear(120, 84)
        # input dim = 84, output dim = 10
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        #x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


if __name__ == "__main__":
    from Assignment8.Code import kDataPath
    import Assignment5Support

    (xRaw, yRaw) = Assignment5Support.LoadRawData(kDataPath, includeLeftEye=True, includeRightEye=True)

    (xTrainRaw, yTrainRaw, xTestRaw, yTestRaw) = Assignment5Support.TrainTestSplit(xRaw, yRaw, percentTest=.25)

    print("Train is %f percent closed." % (sum(yTrainRaw) / len(yTrainRaw)))
    print("Test is %f percent closed." % (sum(yTestRaw) / len(yTestRaw)))

    from PIL import Image
    import torchvision.transforms as transforms
    import torch
    from EvaluationsStub import Evaluation

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

    for i in range(500):
        # Do the forward pass
        yTrainPredicted = model(xTrain)

        # Compute the training set loss
        loss = lossFunction(yTrainPredicted, yTrain)
        print(i, loss.item())

        # Reset the gradients in the network to zero
        optimizer.zero_grad()

        # Backprop the errors from the loss on this iteration
        loss.backward()

        # Do a weight update step
        optimizer.step()

    yTestPredicted = model(xTest)

    yPred = [1 if pred > 0.5 else 0 for pred in yTestPredicted]

    ev = Evaluation(yTest, yPred)
    # print("Accuracy simple:", Evaluations.Accuracy(yTest, yPred))
    # simpleAccuracy = Evaluations.Accuracy(yTest, yPred)
    print("Accuracy simple:", ev.accuracy)
    simpleAccuracy = ev.accuracy