import unittest
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision.utils import make_grid

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(root_dir, "src")
sys.path.insert(0, src_dir)
# change cwd to root dir
os.chdir(root_dir)

from data_loader.CaptchaDataLoader import CaptchaDataLoader

class CaptchaDataLoaderTestCase(unittest.TestCase):

    def preview(self, dataset):
        kwargs = {"nrow": 4, "padding": 40}
        for i, (_, images, labels, input_lengths, target_lengths) in enumerate(dataset):
            if i == 0:
                for j in labels:
                    print(''.join([dataset.dataset.characters[x] for x in j]))
                image = make_grid(images, pad_value=-1, **kwargs).numpy()
                image = np.transpose(image, (1, 2, 0))
                plt.figure()
                plt.imshow(image)
                plt.show()
                return

    def test_training(self):
        dataset = CaptchaDataLoader("", 16, characters=" 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()",
        n_len=[1,5])
        self.preview(dataset)


if __name__ == '__main__':
    unittest.main()
