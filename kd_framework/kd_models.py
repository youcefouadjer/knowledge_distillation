import torch
import torch.nn as nn
import torch.nn.functional as F


def l2_norm(input_tensor, axis=1, eps=1e-12):
    norm = torch.norm(input_tensor, p=2, dim=axis, keepdim=True)
    return input_tensor / (norm + eps)

class TeacherNet(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base = base_model
        self.classifier = nn.Linear(self.base.last_bn.num_features, num_classes)

    def forward(self, x):
        x = self.base.conv2d_1a(x)
        x = self.base.conv2d_2a(x)
        x = self.base.conv2d_2b(x)
        x = self.base.maxpool_3a(x)
        x = self.base.conv2d_3b(x)
        x = self.base.conv2d_4a(x)
        x = self.base.conv2d_4b(x)
        x = self.base.repeat_1(x)
        feat = x  # intermediate feature

        x = self.base.mixed_6a(x)
        x = self.base.repeat_2(x)
        x = self.base.mixed_7a(x)
        x = self.base.repeat_3(x)
        x = self.base.block8(x)
        x = self.base.avgpool_1a(x)
        x = self.base.dropout(x)
        x = self.base.last_linear(x.view(x.shape[0], -1))
        x = self.base.last_bn(x)
        embedding = F.normalize(x, p=2, dim=1)
        logits = self.classifier(x)

        return logits, [feat]



class StudentNet(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base = base_model
        self.classifier = nn.Linear(self.base.bn.num_features, num_classes)

    def forward(self, x):
        x = self.base.conv1(x)
        x = self.base.conv2_dw(x)
        x = self.base.conv_23(x)
        x = self.base.conv_3(x)
        x = self.base.conv_34(x)
        feat = self.base.conv_4(x)  # feature pour distillation
        x = self.base.conv_45(feat)
        x = self.base.conv_5(x)
        x = self.base.conv_6_sep(x)
        x = self.base.conv_6_dw(x)
        x = self.base.conv_6_flatten(x)
        x = self.base.linear(x)
        x = self.base.bn(x)

        embedding = l2_norm(x)          # 🔹 Pour vérification
        logits = self.classifier(x)     # 🔹 Pour classification

        return logits, [feat], embedding  # 🔹 Pour distillation

