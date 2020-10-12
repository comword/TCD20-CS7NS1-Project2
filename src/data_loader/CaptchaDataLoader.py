from base import BaseDataLoader
import torch
import os.path
import random
import numpy as np

from .CaptchaDataset import CaptchaDataset
from .ImageDataset import ImageDataset
from model.CaptchaCNN import CaptchaCNN


class CaptchaDataLoader(BaseDataLoader):

    def __init__(self, data_path, batch_size, n_len=[5,6], train_total=5000, width=128, height=64, 
                 shuffle=True, validation_split=0.0, num_workers=1, training=True,
                 characters=' 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        self.characters = characters
        self.width = width
        self.height = height
        self.n_len = n_len
        self.train_total = train_total
        self.data_path = data_path

        model = CaptchaCNN(len(characters), input_shape=(3, self.height, self.width))
        inputs = torch.zeros((1, 3, self.height, self.width))
        outputs = model(inputs)

        self.input_length = outputs.shape[0]
        if training:
            self.dataset = CaptchaDataset(self.characters, self.train_total, self.width, self.height, self.input_length, self.n_len)
        else:
            self.dataset = ImageDataset(data_path)
            self.length, self.width, self.height = self.dataset.__len__(), self.dataset._width, self.dataset._height
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

    def get_dataset(self):
        return self.dataset
