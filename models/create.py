import torch
from models import Yolo
from models import YoloTiny
from models import YoloTinyMS

def create_model(weight_path, S, B, num_classes):
    model = YoloTiny(S, B, num_classes)
    # load pretrained model
    if weight_path and weight_path != '':
        model.load_state_dict(torch.load(weight_path, map_location="cuda:0"))
    return model
