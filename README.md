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
The dataset can be used in the following way
```python
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
from torchvision.transforms import transforms

from omniart_eye_dataset import OmniArtEyeDataset

dataset = OmniArtEyeDataset(transform=transforms.Compose([
                               transforms.Resize(50),
                               transforms.CenterCrop(50),
                               transforms.ToTensor(),
                           ]))
                           
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
This dataset has already been used to train a [classifier](https://github.com/rogierknoester/omniart_eye_classifier) and painted eye generator.



### Project origin
This package is part of a Master's thesis at the University of Amsterdam.