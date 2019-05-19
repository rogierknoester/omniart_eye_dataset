import glob
import os
from typing import Tuple, Union
import subprocess

import pandas as pd
from PIL.Image import Image
from torch import Tensor

from torchvision.datasets import ImageFolder

OA_DATASET_FULL = 'OA_DATASET_FULL'
OA_DATASET_COLOR = 'OA_DATASET_COLOR'
OA_DATASET_FULL_25x25 = 'OA_DATASET_FULL_25x25'
OA_DATASET_COLOR_25x25 = 'OA_DATASET_COLOR_25x25'


class OmniArtEyeDataset(ImageFolder):
    dataset_tar = {
        OA_DATASET_FULL: 'omniart_eye_dataset.tar.xz',
        OA_DATASET_COLOR_25x25: 'omniart_eye_dataset_color_25x25.tar.xz'
    }
    dataset_tar_part = {
        OA_DATASET_FULL: 'omniart_eye_dataset.tar.*'
    }
    metadata_tar = 'omniart_metadata.tar.xz'

    def __init__(self, transform=None, dataset_type=OA_DATASET_FULL):
        self.dataset_type = dataset_type
        self.transform = transform
        self.root = os.path.join(os.path.dirname(__file__), 'datasets')

        if self.dataset_type == OA_DATASET_FULL:
            self.dataset_folder = 'full'
        elif self.dataset_type == OA_DATASET_COLOR_25x25:
            self.dataset_folder = 'color_25x25'
        else:
            raise ValueError('Unable to use dataset type %s, it is not implemented'.format(dataset_type))

        if not self.__data_files_exist():
            self.__unpack_datasets()

        self.__csv = pd.read_csv(os.path.join(self.root, 'omniart_metadata.csv'), low_memory=False)

        super(OmniArtEyeDataset, self).__init__(os.path.join(self.root, self.dataset_folder), transform=self.transform)

    def __data_files_exist(self) -> bool:
        return os.path.isdir(os.path.join(self.root, self.dataset_folder)) and \
               os.path.isfile(os.path.join(self.root, 'omniart_metadata.csv'))

    def __unpack_datasets(self):
        import tarfile
        print("Unpacking OmniArt eyes dataset...")

        # extract the dataset
        if self.dataset_type == OA_DATASET_FULL:
            # First join the part files
            # Part files exist because of file size limits and the lack of an external (permanent) hosting for the files
            part_files = glob.glob(os.path.join(self.root, self.dataset_tar_part[self.dataset_type]))
            with open(os.path.join(self.root, self.dataset_tar[self.dataset_type]), 'wb') as dataset_file:
                for part_file in part_files:
                    with open(part_file, 'rb') as read_file:
                        dataset_file.write(read_file.read())

        with tarfile.open(os.path.join(self.root, self.dataset_tar[self.dataset_type])) as tar:
            tar.extractall(path=self.root)
        print("Unpacking OmniArt metadata...")
        with tarfile.open(os.path.join(self.root, self.metadata_tar)) as tar:
            tar.extractall(path=self.root)

    def __getitem__(self, index: int) -> Tuple[Union[Image, Tensor], int, dict]:
        image, color = super().__getitem__(index)
        omni_id = self.__get_omni_id(index)

        return image, color, self.__get_omniart_metadata(omni_id)

    def __get_omniart_metadata(self, omni_id: int) -> dict:
        # There should only exist one row with a given omni id, thus we can safely assume we receive 1 row
        return self.__csv.loc[self.__csv['omni_id'] == omni_id].to_dict(orient='records')

    def __get_omni_id(self, index: int) -> int:
        filename = os.path.basename(self.samples[index][0])
        return int(filename.split('_')[0])
