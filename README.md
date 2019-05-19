# OmniArt Eye Dataset
This package provides a dataset of 118,576 painted eyes. These eyes are extracted from about 245,000 paintings from the [OmniArt](http://isis-data.science.uva.nl/strezoski/#2) dataset.
The dataset provides the eyes as images, the colour of the iris, and metadata from the OmniArt dataset.


#### Usage
The dataset can be used like any other PyTorch dataset. It extends the ``ImageFolder`` class to provide the images and labels/colour and in addition attaches the OmniArt metadata as a dictionary.
The following classes are used, and how many of that class exist in the dataset:  

| Colour    | Count |
|-----------|-------|
| Amber     | 3,114  |
| Blue      | 18,926 |
| Brown     | 42,094 |
| Gray      | 7,637  |
| Grayscale | 16,234 |
| Green     | 5,220  |
| Hazel     | 1,578  |
| Irisless  | 18,282 |
| Negative  | 96,042 |
| Red       | 5,491  |

The ``negative`` class exists to be able to classify non-eye images. It contains samples of primarily noise and facial areas, such as closed eyelids.

###### Example
The dataset can be used in the following way. It is possible to specify which (sub)dataset to use. The full dataset, the colour only dataset (excludes the irisless and negative classes), and whether to use all size images or only 25 by 25 and higher resolutions.
```python
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
from torchvision.transforms import transforms

from omniart_eye_dataset import OmniArtEyeDataset, OA_DATASET_COLOR_25x25, OA_DATASET_FULL

dataset = OmniArtEyeDataset(transform=transforms.Compose([
    transforms.Resize(50),
    transforms.CenterCrop(50),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]), dataset_type=OA_DATASET_COLOR_25x25)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

# Take 64 random entries
images, color, metadata = next(iter(dataloader))

# Plot the entries
plt.figure(figsize=(10, 10))
plt.axis("off")
plt.title("OmniArt eyes")
plt.imshow(np.transpose(vutils.make_grid(images, padding=5, normalize=True), (1, 2, 0)))
plt.show()
```

![Sample eyes](images/sample.png?raw=true)  





### Related
This dataset has already been used to train a [classifier](https://github.com/rogierknoester/omniart_eye_classifier) and [painted eye generator](https://github.com/rogierknoester/omniart_eye_generator).



### Project origin
This package is part of a Master's thesis at the University of Amsterdam.