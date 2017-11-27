import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc0 = nn.Conv2d(1, 2, 5)
        self.fc1 = nn.Conv2d(2, 4, 5)
        self.fc2 = nn.Conv2d(4, 8, 5)
        self.fc3 = nn.MaxPool2d(2)
        self.fc4 = nn.Linear(512, 27)

    def forward(self, x):
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
#        print(x.size())
 #       import ipdb;ipdb.set_trace()
        return F.log_softmax(x)
