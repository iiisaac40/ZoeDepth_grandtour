# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os

from PIL import Image
import numpy as np
import cv2


class ToTensor(object):
    def __init__(self):
        # self.normalize = transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.normalize = lambda x : x
        # self.resize = transforms.Resize((375, 1242))

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        image = self.to_tensor(image)
        image = self.normalize(image)
        depth = self.to_tensor(depth)

        # image = self.resize(image)

        return {'image': image, 'depth': depth, 'dataset': "grandtour"}

    def to_tensor(self, pic):

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        #         # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()

        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img



class GRANDTOUR(Dataset):
    def __init__(self, data_dir_root, accumulate_level='100', do_kb_crop=True):
        self.data_dir_root = data_dir_root
        # image paths are of the form <data_dir_root>/<camera name>/*.png
        # depth image paths are of the form <data_dir_root>/depth/<accumulate level>/<camera name>/*.png
        filelist_path = os.path.join(data_dir_root, 'txt_files', f"test_{accumulate_level}_files.txt")
        with open(filelist_path, 'r') as f:
            self.filelist = f.read().splitlines()
            # self.filelist.sort()
        
        self.transform = ToTensor()
        self.do_kb_crop = True

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir_root, 'GrandTour', self.filelist[idx].split(' ')[0])
        depth_path = os.path.join(self.data_dir_root, 'GrandTour', self.filelist[idx].split(' ')[1])
        

        image = Image.open(image_path)
        depth = Image.open(depth_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR |
                           cv2.IMREAD_ANYDEPTH) / 1000.0
        depth = Image.fromarray(depth)

        # if self.do_kb_crop:
        #     if idx == 0:
        #         print("Using KB input crop")
        #     height = image.height
        #     width = image.width
        #     top_margin = int(height - 352)
        #     left_margin = int((width - 1216) / 2)
        #     depth = depth.crop(
        #         (left_margin, top_margin, left_margin + 1216, top_margin + 352))
        #     image = image.crop(
        #         (left_margin, top_margin, left_margin + 1216, top_margin + 352))
        #     # uv = uv[:, top_margin:top_margin + 352, left_margin:left_margin + 1216]

        image = np.asarray(image, dtype=np.float32) / 255.0
        # depth = np.asarray(depth, dtype=np.uint16) /1.
        depth = np.asarray(depth, dtype=np.float32) / 1.
        depth[depth > 80] = -1
        print("gt dpeth min max", depth.min(), depth.max())

        depth = depth[..., None]
        sample = dict(image=image, depth=depth)

        # return sample
        sample = self.transform(sample)

        if idx == 0:
            print(sample["image"].shape)

        sample['image_path'] = image_path

        return sample

    def __len__(self):
        return len(self.filelist)


def get_grandtour_loader(data_dir_root, accumulate_level, batch_size=1, **kwargs):
    dataset = GRANDTOUR(data_dir_root, accumulate_level, )
    return DataLoader(dataset, batch_size, **kwargs)


if __name__ == "__main__":
    loader = get_grandtour_loader(
        data_dir_root="home/grand_tour_depth_benchmark/evaluation/datasets/")
    print("Total files", len(loader.dataset))
    for i, sample in enumerate(loader):
        print(sample["image"].shape)
        print(sample["depth"].shape)
        print(sample["dataset"])
        print(sample['depth'].min(), sample['depth'].max())
        if i > 5:
            break
