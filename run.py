import argparse
import os
import sys
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tqdm import tqdm

src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, src_dir)

from utils import prepare_device
from parse_config import ConfigParser
import model as module_arch
from data_loader.CaptchaDataLoader import CaptchaDataLoader


class DemoMain:
    def __init__(self, config):
        self.config = config

        self.logger = config.get_logger('demo')
        self.device, device_ids = prepare_device(self.logger, config['n_gpu'])

        torch.set_grad_enabled(False)
        self.model = config.init_obj('arch', module_arch)
        self.logger.info('Loading checkpoint: {} ...'.format(config.resume))
        if config['n_gpu'] > 0:
            if torch.cuda.is_available():
                checkpoint = torch.load(config.resume)
            else:
                checkpoint = torch.load(config.resume, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(config.resume, map_location=torch.device('cpu'))
        state_dict = checkpoint['state_dict']
        if config['n_gpu'] > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def decode(sequence, characters):
        a = ''.join([characters[x] for x in sequence])
        s = ''.join([x for j, x in enumerate(a[:-1]) if x != characters[0] and x != a[j+1]])
        if len(s) == 0:
            return ''
        if a[-1] != characters[0]:
            s += a[-1]
        return s

    def main(self, image):
        self.data_loader = CaptchaDataLoader(image, 16, training=False, shuffle=False)
        self.characters = self.config["data_loader"]["args"]["characters"]
        tbar = tqdm(self.data_loader)
        with torch.no_grad():
            with open("stuff.txt", "w") as f:
                for _, (imgid, images, _, _, _) in enumerate(tbar):
                    images = images.to(self.device)
                    output = self.model(images)
                    output_argmax = output.detach().permute(1, 0, 2).argmax(dim=-1)
                    output_argmax = output_argmax.cpu().numpy()
                    for res in zip(imgid, output_argmax):
                        f.write("%s, %s\n" % (res[0], self.decode(res[1], self.characters)))
                
if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Demo')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-i', '--image', type=str, required=True,
                      help='path to image to be processed')

    config = ConfigParser.from_args(args)
    config.init_log()
    m = DemoMain(config)
    arg_parsed = args.parse_args()
    m.main(arg_parsed.image)
