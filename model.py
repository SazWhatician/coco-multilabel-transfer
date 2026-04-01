# model.py
import torch
import torch.nn as nn
import torchvision.models as models

VOC_CLASSES = [
    "aeroplane","bicycle","bird","boat","bottle",
    "bus","car","cat","chair","cow",
    "diningtable","dog","horse","motorbike","person",
    "pottedplant","sheep","sofa","train","tvmonitor"
]
NUM_CLASSES = len(VOC_CLASSES)
IMG_SIZE    = 320


class MultiLabelResNet(nn.Module):
    """
    ResNet50 pretrained backbone + custom multi-label head
    """
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.5):
        super().__init__()

        backbone = models.resnet50(weights=None)

        self.stage1 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.stage2 = backbone.layer1
        self.stage3 = backbone.layer2
        self.stage4 = backbone.layer3
        self.stage5 = backbone.layer4

        self.gap  = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.gap(x)
        return self.head(x)



MultiLabelVOCNet = MultiLabelResNet


def load_model(weights_path: str, device: str = "cpu") -> MultiLabelResNet:
    model = MultiLabelResNet(num_classes=NUM_CLASSES)
    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print(f"Loaded weights from {weights_path}")
    return model