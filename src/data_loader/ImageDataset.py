import numpy as np
import torch.utils.data as data
import os
from pathlib import Path
from PIL import Image
from torchvision.transforms.functional import to_tensor

class ImageDataset(data.Dataset):
    """ImageDataset.
        Args:
            root (string): Root directory of dataset.
            train (bool, optional): If True, creates dataset from ``training``,
                otherwise from ``test``.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
        """
    files = []

    def __init__(self, root, train=True, transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train  # training set or test set
        self.files = []
        self._set_files()

    def _set_files(self):
        """
        Create a file path/image id list.
        """
        self.files = [file for file in os.listdir(self.root) if file.endswith('.png')]

    def _load_data(self, image_id):
        """
        Load the image and label in numpy.ndarray
        """
        image_id = self.files[image_id]
        image_path = os.path.join(self.root, image_id)
        image = Image.open(image_path)
        # return os.path.basename(self.root) + "_" + os.path.splitext(image_id)[0], image
        return image_id, image

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        """
            Args:
            index (int): Index
            Returns:
            tuple: (image_id, image, target)
        """
        image_id, image = self._load_data(index)

        if self.transform is not None:
            image = self.transform(image)
        
        image = np.asarray(image)
        return image_id, to_tensor(image), [], [], []

    def __repr__(self):
        fmt_str = "Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    # data: {}\n".format(self.__len__())
        fmt_str += "    Root: {}".format(self.root)
        return fmt_str

    @property
    def _width(self):
        image_id, image = self._load_data(0)
        return image.size[0]

    @property
    def _height(self):
        image_id, image = self._load_data(0)
        return image.size[1]

if __name__ == "__main__":
    imgset = ImageDataset("data/geto-project1")
    image_id, image = imgset._load_data(0)
    print(image_id, image.size)
