from base import BaseDataLoader
import torch
import torch.nn.utils.rnn as rnn_utils
import os.path
import random
import numpy as np
from PIL import Image

from .CaptchaDataset import CaptchaDataset
from .ImageDataset import ImageDataset
from model.CaptchaCNN import CaptchaCNN


class CaptchaDataLoader(BaseDataLoader):

    def __init__(self, data_path, batch_size, n_len=[5,6], train_total=5000, resize_to=[128, 64], 
                 gen_size=[128, 64], shuffle=True, validation_split=0.0, num_workers=0, training=True,
                 characters=' 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        self.characters = characters
        self.resize_to = resize_to
        self.gen_size = gen_size
        self.width = resize_to[0]
        self.height = resize_to[1]
        self.n_len = n_len
        self.train_total = train_total
        self.data_path = data_path

        model = CaptchaCNN(len(characters), input_shape=(3, self.height, self.width))
        inputs = torch.zeros((1, 3, self.height, self.width))
        outputs = model(inputs)

        self.input_length = outputs.shape[0]
        if training:
            self.dataset = CaptchaDataset(self.characters, self.train_total, self.gen_size, self.resize_to, self.input_length, self.n_len)
        else:
            self.dataset = ImageDataset(data_path, transform=self.trsfm)
            self.length = self.dataset.__len__()
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

    def get_dataset(self):
        return self.dataset

    def trsfm(self, image):
        # w, h = image.size
        return self._augmentation(image, self.width, self.height)

    def _augmentation(self, img, w, h):
        img = img.resize((w, h), resample=Image.NEAREST)
        return img