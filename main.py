import matplotlib.pyplot as plt  # for plotting
import numpy as np  # for transformation

import torch  # PyTorch package
import torchvision  # load datasets
import torchvision.transforms as transforms  # transform data
import torch.nn as nn  # basic building block for neural neteorks
import torch.nn.functional as F  # import convolution functions like Relu
import torch.optim as optim  # optimzer

# python image library of range [0, 1]
# transform them to tensors of normalized range[-1, 1]
import imageio
import glob
from load_data import load_data


if __name__ == '__main__':

    batch_size = 64
    trainloader, classes = load_data()
    # data_iter = iter(train_loader)
    # images = data_iter.__next__()

    def imshow(img):
        npimg = img.numpy()  # convert to numpy objects
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

        ''' function to show image '''
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()  # convert to numpy objects
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


    # get random training images with iter function
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # call function on our images
    imshow(torchvision.utils.make_grid(images))

    # print the class of the image

    print(' '.join('%s' % classes[labels[j]] for j in range(batch_size)))


    class Net(nn.Module):
        ''' Models a simple Convolutional Neural Network'''

        def __init__(self):
            ''' initialize the network '''
            super(Net, self).__init__()
            # 3 input image channel, 6 output channels,
            # 5x5 square convolution kernel
            self.conv1 = nn.Conv2d(3, 6, 5)
            # Max pooling over a (2, 2) window
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 from image dimension
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            ''' the forward propagation algorithm '''
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x



    net = Net()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Assuming that we are on a CUDA machine, this should print a CUDA device:

    print(device)
    # USING CUDA EHHEHEEHEHEHEHHEHEHEHHEH
    net.to(device)

    print(net)

    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

    # net = Net()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Assuming that we are on a CUDA machine, this should print a CUDA device:

    print(device)
    # USING CUDA EHHEHEEHEHEHEHHEHEHEHHEH
    net.to(device)

    net.load_state_dict(torch.load(PATH))

    outputs = net(images)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                                  for j in range(4)))

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


    inputs, labels = data[0].to(device), data[1].to(device)
