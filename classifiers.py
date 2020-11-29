import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors


class KNN:
    def __init__(self, train_set, val_set):

        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=len(train_set),
                                                   shuffle=True,
                                                   num_workers=25,
                                                   drop_last=True)

        val_loader = torch.utils.data.DataLoader(val_set,
                                                 batch_size=len(val_set),
                                                 shuffle=False,
                                                 num_workers=25,
                                                 drop_last=True)

        it_train = iter(train_loader)
        it_test = iter(val_loader)

        self.train_images, self.train_labels, self.train_paths = it_train.next()
        self.test_images, self.test_labels, self.test_paths = it_test.next()

        self.train_size = len(self.train_images)
        self.test_size = len(self.test_images)

        self.train_images = np.squeeze(self.train_images.numpy().reshape((self.train_size, 3*128*128)))
        self.test_images = np.squeeze(self.test_images.numpy().reshape((self.test_size, 3*128*128)))
        self.model = None

    def train(self):
        self.model = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(self.train_images)

    def test(self):
        # returns instances and indices
        print(self.model.kneighbors(self.test_images))
        # do K means with no loop
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
