import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim

from classes import get_classes
from config import *
from dataset import PoseDataset
from network import Net

def main():
    classes = get_classes()
    net = Net(len(classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    print('loading training dataset...')
    train_set = PoseDataset(training_dir, classes)
    print('loading validation dataset...')
    validation_set = PoseDataset(validation_dir, classes)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle= True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=1, shuffle= True)

    best_validation_loss = 100
    for epoch in range(num_epochs):
        running_loss = 0.0
        # train
        for data in train_loader:
            input, label = data
            optimizer.zero_grad()
            output = net(input)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

        # validate
        validation_loss = 0.0
        for data in validation_loader:
            input, label = data
            with torch.no_grad():
                output = net(input)
                loss = criterion(output, label)
                validation_loss += loss.item()
        validation_loss /= len(validation_loader)
        print(f'epoch {epoch}: validation_loss = {validation_loss:.3f}')
        if validation_loss < best_validation_loss:
            print('saving')
            torch.save(net.state_dict(), 'best_model.pt')
            best_validation_loss = validation_loss


if __name__ == '__main__':
    main()
