import torch
from models import YoloTinier
from models import YoloTiny
from models import YoloMS

def create_model(weight_path, S, B, num_classes, model_type='ms', device='cpu'):
    if model_type == 'ms':
        model = YoloMS(S, B, num_classes)
    elif model_type == 'tiny':
        model = YoloTiny(S, B, num_classes)
    elif model_type == 'tinier':
        model = YoloTinier(S, B, num_classes)
    # load model, if weight exist
    if weight_path and weight_path != '':
        model.load_state_dict(torch.load(weight_path, map_location=device))
    return model
