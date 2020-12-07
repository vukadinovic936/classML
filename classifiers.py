""" In this file we store all the classifers used for the project """
import time
import copy
import torch
from torchvision import models
from tqdm import tqdm
import numpy as np
from sklearn.neighbors import NearestNeighbors


class KNN:
    """ K nearest neighbour classifier
    this is very inefficient way for classification
    but we create it for the sake of showing how advanced CNNs are
    this class is made to accept data loaders so they have to have all
    images in 1 batch
    """
    def __init__(self, train_loader, val_loader, train_size, val_size):

        it_train = iter(train_loader)
        it_test = iter(val_loader)

        self.train_images, \
            self.train_labels, \
            self.train_paths = it_train.next()

        self.test_images, self.test_labels, self.test_paths = it_test.next()

        self.train_size = len(self.train_images)
        self.test_size = len(self.test_images)

        self.train_images = np.squeeze(self.train_images.numpy()
                                       .reshape((self.train_size,
                                                 3*128*128)))

        self.test_images = np.squeeze(self.test_images.numpy()
                                      .reshape((self.test_size,
                                                3*128*128)))
        self.model = None

    def train(self):
        """ Train phase of KNN using ball_tree algorithm """
        self.model = NearestNeighbors(n_neighbors=2, algorithm='ball_tree')\
            .fit(self.train_images)

    def test(self):
        """ Testing phase
        should return accuracy """
        # returns instances and indices
        out = self.model.kneighbors(self.test_images)
        first_nb = out[1][:, 0]

        for i in range(len(first_nb)):
            first_nb[i] = self.train_labels[first_nb[i]]

        unique, counts = np.unique((self.test_labels.numpy() == first_nb),
                                   return_counts=True)
        res = dict(zip(unique, counts))
        return res[True]/(res[False]+res[True])


class CNN:
    """ Convolutional Neural Networks using transfered learning
    from resnet 18 architecture """
    def __init__(self, train_loader, val_loader, train_size, val_size):

        self.dataloaders = {'train': train_loader, 'val': val_loader}
        self.dataset_sizes = {'train': train_size, 'val': val_size}
        # if you don't have a GPU available please change this
        self.device = torch.device("cuda:0")
        self.model = models.resnet18(pretrained=True)
        num_fltrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_fltrs, 196)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=0.001,
                                         momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=7,
                                                         gamma=0.1)
        self.num_epochs = 25
        self.model.to(self.device)

    def train(self):
        """ Training phase of CNN
        we measure time
        we measure accuracy through epoch
        model with the best accuracy is saved and returned"""
        since = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in tqdm(range(self.num_epochs)):
            print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            print('-' * 10)
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels, _ in tqdm(self.dataloaders[phase]):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    self.scheduler.step()

                epoch_loss = running_loss / self.dataset_sizes[phase]

                epoch_acc = \
                    running_corrects.double() / self.dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    torch.save(self.model.state_dict(), "best/model.pth")

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        return self.model

    def test(self):
        """ Returns the accuracy of the best performing model of CNN"""

        for phase in ['val']:
            self.model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels, _ in tqdm(self.dataloaders[phase]):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = self.criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                self.scheduler.step()

            epoch_loss = running_loss / self.dataset_sizes[phase]

            epoch_acc = \
                running_corrects.double() / self.dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

        print()
        return epoch_acc
