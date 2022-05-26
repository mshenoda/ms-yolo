import torch
from torch import nn
import torchvision.transforms.functional as F

class RandomBlur(nn.Module):
    def __init__(self, kernel_size=[3,3], sigma=[0.1, 2.0], p=0.5):
        super().__init__()
        self.p = p
        self.sigma = sigma
        self.kernel_size = kernel_size
    
    def get_params(self, sigma_min: float, sigma_max: float) -> float:
        return torch.empty(1).uniform_(sigma_min, sigma_max).item()

    def forward(self, img):
        if torch.rand(1) < self.p:
            sigma = self.get_params(self.sigma[0], self.sigma[1])
            return F.gaussian_blur(img, self.kernel_size, [sigma, sigma])
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"

class RandomHorizontalFlip(nn.Module):
    def __init__(self, probability=0.5):
        super().__init__()
        self.p = probability

    def forward(self, img, bboxes):
        # bbox format is x, y, w, h, c
        if torch.rand(1) < self.p:
            img = F.hflip(img)
            result_bboxes = []
            for x, y, w, h, c in bboxes:
                result_bboxes.append((w-x, y, w, h, c))
            return (img, result_bboxes)
        return (img, bboxes)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"

class RandomVerticalFlip(nn.Module):
    def __init__(self, probability=0.5):
        super().__init__()
        self.p = probability

    def __call__(self, img, bboxes):
        # bbox format is x, y, w, h, c
        if torch.rand(1)  < self.p:
            img = F.vflip(img)
            result_bboxes = []
            for x, y, w, h, c in bboxes:
                result_bboxes.append((x, h-y, w, h, c))
            return (img, result_bboxes)
        return (img, bboxes)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"