import json
import os

import numpy as np
from PIL import Image

import grain.python as grain


def load_json_file(json_file: str) -> list[dict[str, str]]:
    """load the json file that contains the paths of samples

    Args:
        json_file: the absolute path to the json file

    Returns:
        data: a list of dictionary, where each contains the path of each sample
    """
    with open(file=json_file, mode='r') as f:
        data = json.load(fp=f)
    
    return data


class ImageDataSource(grain.RandomAccessDataSource):
    def __init__(self, json_file: str, root: str = None) -> None:
        """make the dataset from multiple annotation files.

        Each file may contain only a subset of the whole dataset.
        If one annotator does not label a sample, the label will be set to -1.

        Args:
            json_file: path to the json file of ground truth
            root: the directory to the dataset folder

        Returns:
            dataset:
        """
        self.root = root if root is not None else ''
        self._data = load_json_file(json_file=json_file)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        """
        """
        # load images
        x = Image.open(fp=os.path.join(self.root, self._data[idx]['file']))
        x = np.array(object=x)

        y = np.array(object=self._data[idx]['label'], dtype=np.int32)
        if len(x.shape) != 3:
            x = np.stack(arrays=[x, x, x], axis=-1)
            

        return dict(filename=self._data[idx]['file'], image=x, label=y)

    def __len__(self) -> int:
        return len(self._data)