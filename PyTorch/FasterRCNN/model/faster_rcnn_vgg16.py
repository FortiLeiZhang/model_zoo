import torch as t
from torch import nn
from torchvision.models import vgg16

def build_vgg16():
    model = vgg16()
    model.load_state_dict(t.load('/home/lzhang/model_zoo/PyTorch/FasterRCNN/pretrained/vgg16-397923af.pth'))
    
    features = list(model.features)[:30]
    classifier = list(model.classifier)
    
    del classifier[6]
    del classifier[5]
    del classifier[2]

    for layer in features[:10]:
        for p in layer.parameters():
            p.require_grad = False
            
    return nn.Sequential(*features), nn.Sequential(*classifier)

if __name__ == '__main__':
    build_vgg16()