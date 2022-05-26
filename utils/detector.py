import cv2
from torchvision import transforms
from PIL import Image
from utils import decode
from models import create_model

class Detector(object):
    def __init__(self, model_path, input_size, S, B, num_classes):
        super().__init__()
        self.model = create_model(model_path, S, B, num_classes)
        self.transforms = transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor()
        ])
        self.S = S
        self.B = B
        self.num_classes = num_classes

    def detect(self, img, conf, iou):
        image = Image.fromarray(img).convert('RGB')
        image = self.transforms(image)
        image.unsqueeze_(0)

        pred = self.model(image)[0].detach().cpu()
        
        return decode(pred, self.S, self.B, self.num_classes, conf, iou)