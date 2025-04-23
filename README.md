# Ferienprogramm KI

Ferienprogramm KI der Fachhochschule Südwestfalen für 2023.

## Tag 1: Neuronales Netz für Ziffernerkennung
Siehe [MNIST Notebook](Bildklassifikation/MNIST.ipynb)

### CNN für FashionMNIST
Code für das CNN:

```Python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64*5*5, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64*5*5)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

Code für das ResNet:

```Python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2, 2)
        self.shortcut = nn.Sequential()
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += shortcut  # Residual Connection
        x = F.relu(x)
        return x

class ResNetCNN(nn.Module):
    def __init__(self):
        super(ResNetCNN, self).__init__()
        self.layer1 = ResNetBlock(1, 32)
        self.layer2 = ResNetBlock(32, 64)
        self.layer3 = ResNetBlock(64, 128)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)  # Überprüfe die Dimensionen
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.adaptive_avg_pool2d(x, (3, 3))
        x = x.view(-1, 128 * 3 * 3)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
```

### Training der Bildklassifikation auf dem KI-Cluster der Fachhochschule Südwestfalen

1. Melde Dich sich unter [ki.fh-swf.de/jupyterhub](https://login.ki.fh-swf.de/new-jupyterhub) an. Die Zugangsdaten erhälst Du im Kurs.
2. Klicke **danach** auf diesen [Link](https://login.ki.fh-swf.de/new-jupyterhub/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2Ffhswf%2FFerienkursKI.git&urlpath=lab%2Ftree%2FFerienkursKI.git%2FBildklassifikation%2FMNIST.ipynb&branch=main)


## Tag 2: KI für Snake
Siehe Ordner [SnakeAI](SnakeAI)

## Tag 3: ChatBot zu Pokemon
Siehe Ordner [Chatbot](Chatbot)





[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fhswf/FerienkursKI/blob/main/Bildklassifikation/MNIST.ipynb)
