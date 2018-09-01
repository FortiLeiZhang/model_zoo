import torch as t
from torch import nn
from torchvision.models import vgg16

def _build_vgg16():
    model = vgg16()
    model.load_state_dict(t.load('/home/lzhang/model_zoo/PyTorch/FasterRCNN/pretrained/vgg16-397923af.pth'))
    return model

class FeatureExtractionNetwork(nn.Module):
    def __init__(self):
        super(FeatureExtractionNetwork, self).__init__()
        
        self.vgg16 = _build_vgg16()

        features = list(self.vgg16.features)[:30]
        classifier = list(self.vgg16.classifier)

        del classifier[6]
        del classifier[5]
        del classifier[2]

        for layer in features[:10]:
            for p in layer.parameters():
                p.require_grad = False
                
        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(*classifier)

