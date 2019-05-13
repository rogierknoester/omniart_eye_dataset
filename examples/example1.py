import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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
