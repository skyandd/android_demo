import torch
from torch import nn
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile

from torchvision import models

# model = torchvision.models.mobilenet_v3_small(pretrained=True)
# model.eval()
# example = torch.rand(1, 3, 224, 224)
# traced_script_module = torch.jit.trace(model, example)
# optimized_traced_model = optimize_for_mobile(traced_script_module)
# optimized_traced_model._save_for_lite_interpreter("app/src/main/assets/model.pt")

class MobileNetV2_small7(nn.Module):
    def __init__(self, num_classes, use_pre_trained=True):
        super().__init__()
        self.model = models.mobilenet_v2(use_pre_trained).features[:7]
        self.num_classes = num_classes
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=32, out_features=num_classes, bias=True)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        x = self.avg(x)
        x = torch.squeeze(x)
        x = self.classifier(x)

        return x

model = MobileNetV2_small7(num_classes=3, use_pre_trained=True)
ckp = torch.load('MobileNetV2_small7.pth', map_location=torch.device('cpu'))
model.load_state_dict(ckp)
model.eval()
example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example)
traced_script_module_optimized = optimize_for_mobile(traced_script_module)
traced_script_module_optimized._save_for_lite_interpreter("app/src/main/assets/model.pt")
