import torch

def load_model(model, pretrained=True):
    if pretrained:
        model.load_state_dict(torch.load('path_to_pretrained_model'))
    return model
