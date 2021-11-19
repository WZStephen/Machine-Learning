import os

from torch import autograd
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import random
from resnet import Bottleneck
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from resnet import *
import torch.optim as optim


class Train:
    def __init__(self, data_path = 'data/Imagenet16_train/train_data_batch_1'):
        """
        Init Dataset, Model and others
        """
        self.data_path = data_path
        self.model = ResNet50()
        if torch.cuda.device_count() > 1:
            print("There are ", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)
        elif torch.cuda.device_count() == 1:
            print("There is only one GPU")
        else:
            print("Only use CPU")

        if torch.cuda.is_available():
           self.model.cuda()

    def train(self, epoch=10, batch_size=32, learning_rate=0.001, batch_display=50, save_freq=1):
        self.epoch_num = epoch
        self.batch_size = batch_size
        self.lr = learning_rate

        image_size = 64
        DATA_DIR = 'data/Imagenet16_train/train_data_batch_1'
        X_train = np.load(DATA_DIR, allow_pickle=True)['data']
        X_val = np.load(DATA_DIR, allow_pickle=True)['labels']
        loss_function = nn.CrossEntropyLoss().cuda()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        X_train = np.load(self.data_path, allow_pickle=True)
        train = X_train['data'].tolist()
        label = X_train['labels']
        combined = {}
        # for i in range(len(train)):
        for i in range(len(train)):
            combined[label[i]] = train[i]

        for epoch in range(self.epoch_num):
            epoch_count = 0
            total_loss = 0
            dataloader = DataLoader(X_train['data'], batch_size=self.batch_size, shuffle=True)  # num_workers=8

            for i_batch, sample_batch in enumerate(dataloader):
                # Step.1 Load data and label
                images_batch, labels_batch = sample_batch['image'], sample_batch['label']
                """
                for i in range(images_batch.shape[0]):
                    img_tmp = transforms.ToPILImage()(images_batch[i]).convert('RGB')
                    plt.imshow(img_tmp)
                    plt.pause(0.001)
                """
                input_image = autograd.Variable(images_batch)
                target_label = autograd.Variable(labels_batch)

                # Step.2 calculate loss
                output = self.model(input_image)
                loss = loss_function(output, target_label)
                epoch_count += 1
                total_loss += loss

                # Step.3 Update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Check Result
                if i_batch % batch_display == 0:
                    pred_prob, pred_label = torch.max(output, dim=1)
                    print("Input Label : ", target_label[:4])
                    print("Output Label : ", pred_label[:4])
                    batch_correct = (pred_label == target_label).sum().data[0] * 1.0 / self.batch_size
                    print(
                        "Epoch : %d, Batch : %d, Loss : %f, Batch Accuracy %f" % (epoch, i_batch, loss, batch_correct))




if __name__ == '__main__':
    Train = Train()
    Train.train()

