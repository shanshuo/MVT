import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[1])



class MVDataSet(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='_{:03}.jpg', view_num=6, total_view=12, transform=None,
                 force_grayscale=False, random_shift=True, test_mode=False):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.view_number = view_num
        self.total_view = total_view
        print(f'MVDataSet.view_number =  {self.view_number}')
        print('*****************************************')
        print(f'total view =  {self.total_view}')
        print('*****************************************')

        if self.modality == 'RGBDiff':
            self.new_length += 1# Diff needs one more image to calculate diff
        self._parse_list()

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            return [Image.open(directory + self.image_tmpl.format(idx)).convert('RGB')]

        elif self.modality == 'Flow':
            x_img = Image.open(os.path.join(directory, self.image_tmpl.format('x', idx))).convert('L')
            y_img = Image.open(os.path.join(directory, self.image_tmpl.format('y', idx))).convert('L')
            return [x_img, y_img]

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]

    def _sample_indices(self, record):
        """
        :param record: VideoRecord
        :return: list
        """
        offsets = np.arange(self.view_number)
        return offsets + 1

    def _get_val_indices(self, record):
        offsets = np.arange(self.view_number)
        return offsets + 1

    def _get_test_indices(self, record):
        offsets = np.arange(self.view_number)
        return offsets + 1

    def __getitem__(self, index):
        record = self.video_list[index]

        if not self.test_mode:
            view_indices = np.linspace(1, self.total_view, self.view_number, dtype=int)
        else:
            view_indices = np.linspace(1, self.total_view, self.view_number, dtype=int)

        return self.get(record, view_indices, index)

    def get(self, record, indices, index):
        images = list()
        for view_idx in indices:
            seg_imgs = self._load_image(record.path, view_idx)
            images.extend(seg_imgs)
        process_data1 = self.transform(images)

        return process_data1,record.label

    def __len__(self):
        return len(self.video_list)
