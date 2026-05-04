from __future__ import annotations
import sys
import random

import os
from typing import Any, Callable, Optional, Tuple

from PIL import Image

from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
import torch
import h5py
from time import time
import io
import numpy as np
import json


class VisDial_train(VisionDataset):
    def __init__(
            self,
            root: str,
            split: str = "train",
    ) -> None:
        super(VisDial_train, self).__init__(root)
        self._split = verify_str_arg(split, "split", ("train", "val"))
        self.root = os.path.join('../dialogues/visdial_1.0_%s' % split)
        self._data_path = os.path.join(self.root)

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")


        with open(os.path.join(self._data_path, f'visdial_1.0_{self._split}.json'), "r") as f:
            self._data = json.load(f)

        self._questions = self._data['data']['questions']
        self._answers = self._data['data']['answers']

        self.image_id_to_file_name = self._image_id_to_file_name()
        self.img_feats = h5py.File('../extract_feats/visdial_img_%s.hdf5' % split, 'r')

    def _image_id_to_file_name(self):
        image_id_to_file_name = {}
        if self._split == 'val':
            prefix = 'VisualDialog_val2018_'
            for dialog in self._data['data']['dialogs']:
                image_id = dialog['image_id']
                file_name = prefix + "%012d.jpg" % image_id
                image_id_to_file_name[image_id] = file_name
        else:
            prefix_train = 'COCO_train2014_'
            prefix_val = 'COCO_val2014_'
            for dialog in self._data['data']['dialogs']:
                image_id = dialog['image_id']
                file_name = prefix_train + "%012d.jpg" % image_id
                if os.path.exists('../images/%s' % file_name):
                    image_id_to_file_name[image_id] = file_name
                else:
                    image_id_to_file_name[image_id] = prefix_val + "%012d.jpg" % image_id

        return image_id_to_file_name

    def __len__(self) -> int:
        return len(self._data['data']['dialogs'])

    def __getitem__(self, idx) -> Tuple[Any, Any]:

        data = self._data['data']['dialogs'][idx]
        file_name = self.image_id_to_file_name[data['image_id']]
        img_feat = torch.tensor(self.img_feats[file_name][...]).unsqueeze(0)

        captions = [data['caption']]
        for k in range(10):
            captions.append(self._questions[data['dialog'][k]['question']] + '? ' + self._answers[data['dialog'][k]['answer']])


        return img_feat, captions, file_name

    def _check_exists(self) -> bool:
        return os.path.exists(self._data_path) and os.path.isdir(self._data_path)



class VisDial_test(torch.utils.data.Dataset):
    """ Dataset class for the queries and their targets (dialog and image)"""
    def __init__(self, split=True):
        self.split = split
        with open('../dialogues/VisDial_v1.0_queries_val.json') as f:
            self.queries = json.load(f)
          
    def __len__(self):
        return len(self.queries)

    def __getitem__(self, i):
        if self.split:
            captions = self.queries[i]['dialog']
        else:
            captions = []
            for dialog_length in range(len(self.queries[i]['dialog'])):
                captions.append(', '.join(self.queries[i]['dialog'][:dialog_length + 1]))
        data_id = self.queries[i]['img'].split('/',1)[-1]
        return captions, captions, data_id

