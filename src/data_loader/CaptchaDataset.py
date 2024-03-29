import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor, to_pil_image

from captcha.image import ImageCaptcha
from tqdm import tqdm
import random
import numpy as np
from collections import OrderedDict

class CaptchaDataset(Dataset):
    def __init__(self, characters, length, gen_size, resize_to, input_length, label_length):
        super(CaptchaDataset, self).__init__()
        self.characters = characters
        self.length = length
        self.gen_size = gen_size
        self.resize_to = resize_to
        self.input_length = input_length
        self.label_length = label_length
        self.n_class = len(characters)
        self.generator = ImageCaptcha(width=self.gen_size[0], height=self.gen_size[1])

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        label_length = random.randrange(self.label_length[0], self.label_length[1])
        random_str = ''.join([random.choice(self.characters[1:]) for j in range(label_length)])
        img = self.generator.generate_image(random_str)
        img = img.resize((self.resize_to[0], self.resize_to[1]))
        image = to_tensor(img)
        target = torch.tensor([self.characters.find(x) for x in random_str], dtype=torch.long)
        target = torch.cat((target, torch.zeros(self.label_length[1]-len(random_str), dtype=torch.long)), 0)
        input_length = torch.full(size=(1, ), fill_value=self.input_length, dtype=torch.long)
        target_length = torch.full(size=(1, ), fill_value=label_length, dtype=torch.long)
        return index, image, target, input_length, target_length

if __name__ == "__main__":
    chars = ' abcdefghijklmnopqrstuvwxyz!\"#$%&()*+-:<=>?@[]^_{|}~'
    dataset = CaptchaDataset(chars, 1, [128, 64], [200, 100], 8, (1, 7))
    index, image, target, input_length, target_length = dataset[0]
    print(''.join([chars[x] for x in target]), input_length, target_length)
    print(image.size())
