import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
TF_ENABLE_ONEDNN_OPTS = 0
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


torch.manual_seed(470)
torch.cuda.manual_seed(470)


# training & optimization hyper-parameters
max_epoch = 2
learning_rate = 0.1
batch_size = 32
# train 1562,5 batches 32img ||| 12496 batches 4
device = 'cuda'
###################################### MODIFIED ###########################
our_momentum = 0.9
our_weight_decay = 1e-4
###################################### MODIFIED ###########################

# model hyper-parameters
output_dim = 10

# Boolean value to select training process
training_process = True


data_dir = os.path.join("cifar-10-batches-py", 'my_data')

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class MyOwnClassifier(nn.Module):
    def __init__(self):
        super(MyOwnClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 6 * 6, 288)
        self.fc2 = nn.Linear(288, 144)
        self.fc3 = nn.Linear(144, 10)

        """
        Dimensions
        torch.Size([4, 3, 32, 32])
        torch.Size([4, 6, 30, 30])
        torch.Size([4, 6, 15, 15])
        torch.Size([4, 16, 13, 13])
        torch.Size([4, 16, 6, 6])
        torch.Size([4, 576])
        torch.Size([4, 288])
        torch.Size([4, 144])
        torch.Size([144, 10])
        """

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*6*6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

"""
# For Baseline Model
my_classifier = MyOwnClassifier()
my_classifier = my_classifier.to(device)

# Print your neural network structure
print(my_classifier)

# optimizer = optim.Adam(my_classifier.parameters(), lr=learning_rate)
optimizer = optim.SGD(my_classifier.parameters(), lr=0.001, momentum=0.9)
"""

myModel = MyOwnClassifier()
myModel = myModel.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=myModel.parameters(),
                            lr=0.001,
                            )

"""
tensor = torch.rand(1, 3, 32, 32).to(device)
print(myModel.forward(tensor))
"""

"""
image, label = train_dataset[0]
# label 6
# image tensorsize[3, 32, 32]

names = train_dataset.classes
# ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

class_idx = train_dataset.class_to_idx
# {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

train_dataset.targets
# traindata with indx
"""
image, label = train_dataset[0]
"""
## Visual
plt.imshow(image[2])
plt.show()
"""
names = train_dataset.classes
"""
torch.manual_seed(42)
fig = plt.figure(figsize=(9, 9))
rows, cols = 4, 4
for i in range(1, rows*cols+1):
    random_ind = torch.randint(0, len(train_dataset), size=[1]).item()
    img, label = train_dataset[random_ind]
    fig.add_subplot(rows, cols, i)
    plt.imshow(img[1], cmap="gray")
    plt.title(names[label])
    plt.axis(False)
plt.show()
"""
train_fe_batch, train_labels_fe = next(iter(train_dataloader))
# torch.Size([32, 3, 32, 32]) torch.Size([32])
"""
random_ind = torch.randint(0, len(train_fe_batch), size=[1]).item()
img, label = train_fe_batch[random_ind], train_labels_fe[random_ind]
plt.imshow(img[0], cmap="gray")
plt.title(names[label])
plt.axis(False)
print(f"Image Size: {img.shape}")
print(f"Label: {label}, label size: {label.shape}")
plt.show()
"""

print("initialized")
start = int(input())

if start == 1:
    print("started")
    epochs = 30
    it = 0
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        myModel.train()

        for inputs, labels in train_dataloader:
            it += 1

            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = myModel(inputs)

            loss = nn.CrossEntropyLoss()(logits, labels)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            acc = (logits.argmax(dim=1) == labels).float().mean()

            if it % 2000 == 0:
                print('[epoch:{}, iteration:{}] train loss : {:.4f} train accuracy : {:.4f}'.format(epoch + 1, it,
                                                                                                    loss.item(),
                                                                                                    acc.item()))

        train_losses.append(loss)

        # test phase
        n = 0.
        test_loss = 0.
        test_acc = 0.

        myModel.eval()
        for inputs_test, labels_test in test_dataloader:
            inputs_test = inputs_test.to(device)
            labels_test = labels_test.to(device)

            logits = myModel(inputs_test)

            test_loss += F.cross_entropy(logits, labels_test, reduction="sum").item()
            test_acc += (logits.argmax(dim=1) == labels_test).float().sum().item()
            n += inputs_test.size(0)

        test_loss /= n
        test_acc /= n
        test_losses.append(test_loss)

        print('[epoch:{}, iteration:{}] test_loss : {:.4f} test accuracy : {:.4f}'.format(epoch + 1, it, test_loss,
                                                                                          test_acc))
