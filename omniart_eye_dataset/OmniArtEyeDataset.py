import os
from typing import Tuple, Union

import pandas as pd
from PIL.Image import Image
from torch import Tensor

from torchvision.datasets import ImageFolder


class OmniArtEyeDataset(ImageFolder):
    dataset_tar = 'omniart_eye_dataset.tar.xz'
    metadata_tar = 'omniart_metadata.tar.xz'

    def __init__(self, transform=None):
        self.transform = transform
        self.root = os.path.join(os.path.dirname(__file__), 'datasets')

        if not self.__data_files_exist():
            self.__unpack_datasets()

        self.__csv = pd.read_csv(os.path.join(self.root, 'omniart_metadata.csv'), low_memory=False)

        super(OmniArtEyeDataset, self).__init__(os.path.join(self.root, 'full'), transform=self.transform)

    def __data_files_exist(self) -> bool:
        return os.path.isdir(os.path.join(self.root, 'full')) and \
               os.path.isfile(os.path.join(self.root, 'omniart_eyes_dataset.csv'))

    def __unpack_datasets(self):
        import tarfile
        print("Unpacking OmniArt eyes dataset...")
        # extract the datasets (color and full)
        with tarfile.open(os.path.join(self.root, self.dataset_tar)) as tar:
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
