import torch.nn as nn
from torchvision import models

def reinit_layers(m: nn.Module):
    for mod in m.modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_uniform_(mod.weight, nonlinearity='relu')
            if getattr(mod, "bias", None) is not None:
                nn.init.constant_(mod.bias, 0)

def build_resnet18_head(split_layer_index: int, pretrained: bool = True) -> nn.Module:
    base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1) if pretrained else models.resnet18(weights=None)
    modules = list(base.children())[:split_layer_index]
    head = nn.Sequential(*modules)
    reinit_layers(head); return head

def build_resnet18_server_body(split_layer_index: int, pretrained: bool = True) -> nn.Module:
    base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1) if pretrained else models.resnet18(weights=None)
    modules = list(base.children())[split_layer_index:-2]
    body = nn.Sequential(*modules); return body

def build_local_tail(num_classes: int = 10) -> nn.Module:
    tail = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(), nn.Linear(512, num_classes))
    reinit_layers(tail); return tail

def build_mobilenetv2_student(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1) if pretrained else models.mobilenet_v2(weights=None)
    in_feats = base.classifier[-1].in_features
    base.classifier[-1] = nn.Linear(in_feats, num_classes)
    reinit_layers(base.classifier[-1]); return base

class ProjectionHead(nn.Module):
    def __init__(self, in_dim=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(), nn.Linear(in_dim, out_dim))
        reinit_layers(self.net)
    def forward(self, x): return self.net(x)
