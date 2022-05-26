import torch
from models import YoloTiny

def create_model(weight_path, S, B, num_classes):
    model = YoloTiny(S, B, num_classes)
    # load pretrained model
    if weight_path and weight_path != '':
        model.load_state_dict(torch.load(weight_path))
    return model
