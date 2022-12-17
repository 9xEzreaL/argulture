import torch
import torch.nn.functional as F
import torch.nn as nn
from model import ResNormLayer
from torchvision import models
import pretrainedmodels


_all = [
    'hardnet',
    'efficientnet',
    'densenet',
    'densenet_201',
    'Resnext'
]

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.view(x.data.size(0),-1)


class hardnet(nn.Module):
    def __init__(self, use_meta=False):
        super(hardnet, self).__init__()
        self.use_meta = use_meta
        model = torch.hub.load('PingoLH/Pytorch-HarDNet', 'hardnet85', pretrained=True)
        self.model_f = model.base[:19]
        self.base = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                Flatten(),
                nn.Dropout(0.2))
        if self.use_meta:
            self.base_f = nn.Sequential(nn.Linear(1280+384, 33, bias=True))
            self.meta = nn.Sequential(
                nn.Linear(2, 384),
                nn.ReLU(inplace=True),
                nn.LayerNorm(384),
                ResNormLayer(384),
            )
        else:
            self.base_f = nn.Sequential(nn.Linear(1280, 33, bias=True))


    def forward(self, input, meta=None):
        for layer in self.model_f:
            input = layer(input)
        output_feature = input
        # output_feature = self.model_f(input)
        if self.use_meta:
            meta = meta[:, :2]
            out_meta = self.meta(meta)
            output_feature0 = self.base(output_feature)
            agg_out = torch.cat([output_feature0, out_meta], 1)
            output = self.base_f(agg_out)
        else:
            output = self.base_f(output_feature)

        return output

class efficientnet(nn.Module):
    def __init__(self, use_meta=False):
        super(efficientnet, self).__init__()
        self.use_model = use_meta
        self.model = models.efficientnet_b4(pretrained=True)
        if use_meta:
            self.model.classifier[1] = nn.Linear(1792+384, 33, bias=True)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.meta = nn.Sequential(
                nn.Linear(2, 384),
                nn.ReLU(inplace=True),
                nn.LayerNorm(384),
                ResNormLayer(384),
            )
        else:
            self.model.classifier[1] = nn.Linear(1792, 33, bias=True)


    def forward(self, input, meta=None):
        if meta is not None:
            meta = meta[:, :2]  # no time
            output_feature = self.model.features(input)
            output_feature = self.avgpool(output_feature)
            output_feature = torch.flatten(output_feature, 1)
            out_meta = self.meta(meta)
            agg_out = torch.cat([output_feature, out_meta], 1)
            output = self.model.classifier(agg_out)
        else:
            output = self.model(input)

        return output

class densenet(nn.Module):
    def __init__(self, use_meta=False):
        super(densenet, self).__init__()
        self.use_meta = use_meta
        self.model = models.densenet121(pretrained=True)
        if use_meta:
            in_features = self.model.classifier.in_features + 384
            self.meta = nn.Sequential(
                nn.Linear(2, 384),
                nn.ReLU(inplace=True),
                nn.LayerNorm(384),
                ResNormLayer(384),
            )
        else:
            in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features=in_features, out_features=33, bias=True)


    def forward(self, input, meta=None):
        if self.use_meta:
            meta = meta[:, :2] # no time
            output_feature = self.model.features(input)
            output_feature = F.relu(output_feature, inplace=True)
            output_feature = F.adaptive_avg_pool2d(output_feature, (1, 1))
            output_feature = torch.flatten(output_feature, 1)
            out_meta = self.meta(meta)
            agg_out = torch.cat([output_feature, out_meta], 1)
            output = self.model.classifier(agg_out)
        else:
            output = self.model(input)
        return output

class densenet_201(nn.Module):
    def __init__(self, use_meta=False):
        super(densenet_201, self).__init__()
        self.use_meta = use_meta
        self.model = models.densenet201(pretrained=True)
        if use_meta:
            in_features = self.model.classifier.in_features + 384
            self.meta = nn.Sequential(
                nn.Linear(2, 384),
                nn.ReLU(inplace=True),
                nn.LayerNorm(384),
                ResNormLayer(384),
            )
        else:
            in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features=in_features, out_features=33, bias=True)

    def forward(self, input, meta=None):
        if self.use_meta:
            meta = meta[:, :2] # no time
            output_feature = self.model.features(input)
            output_feature = F.relu(output_feature, inplace=True)
            output_feature = F.adaptive_avg_pool2d(output_feature, (1, 1))
            output_feature = torch.flatten(output_feature, 1)
            out_meta = self.meta(meta)
            agg_out = torch.cat([output_feature, out_meta], 1)
            output = self.model.classifier(agg_out)
        else:
            output = self.model(input)
        return output

# not work
class arc_efficientnet(nn.Module):
    def __init__(self, use_meta=False):
        super(arc_efficientnet, self).__init__()
        self.use_model = use_meta
        self.model = models.efficientnet_b4(pretrained=True)
        if use_meta:
            self.model.classifier[1] = nn.Linear(1792+384, 33, bias=True)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.meta = nn.Sequential(
                nn.Linear(2, 384),
                nn.ReLU(inplace=True),
                nn.LayerNorm(384),
                ResNormLayer(384),
            )
        else:
            self.model.classifier[1] = nn.Linear(1792, 33, bias=True)
        self.class_weight = nn.Parameter(torch.rand(33, 1792+384))
        self.logit_scale = nn.Parameter(torch.rand(()))

    def forward(self, input, meta=None, label=None):
        if meta is not None:
            meta = meta[:, :2]  # no time
            output_feature = self.model.features(input)
            output_feature = output_feature.mean(dim=(2, 3))
            output_feature = output_feature / output_feature.norm(dim=-1, keepdim=True)
            cw = self.class_weight
            cw = cw / cw.norm(dim=-1, keepdim=True)
            out_meta = self.meta(meta)
            agg_out = torch.cat([output_feature, out_meta], 1)
            logits = agg_out @ cw.t()
        else:
            output_feature = output_feature.mean(dim=(2, 3))
            output_feature = output_feature / output_feature.norm(dim=-1, keepdim=True)
            cw = self.class_weight
            cw = cw / cw.norm(dim=-1, keepdim=True)
            logits = output_feature @ cw.t()

        # if label is not None:
        #     # Do the computation for arcface
        #     cos_logits = (1 - logits * logits).clamp(1e-8, 1 - 1e-8) ** 0.5
        #     # one_hot = F.one_hot(label, 33)
        #     logits = torch.where(label == 1,
        #                          0.8 * logits - 0.6 * cos_logits,
        #                          logits)
        return logits * self.logit_scale.exp()

# not work
class PNASNet(nn.Module):
    def __init__(self, use_meta=False):
        super(PNASNet, self).__init__()
        self.use_meta = use_meta
        self.model = pretrainedmodels.pnasnet5large(num_classes=1000, pretrained='imagenet')
        print(self.model)
        assert 0
        if use_meta:
            in_features = self.model.classifier.in_features + 384
            self.meta = nn.Sequential(
                nn.Linear(2, 384),
                nn.ReLU(inplace=True),
                nn.LayerNorm(384),
                ResNormLayer(384),
            )
        else:
            in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features=in_features, out_features=33, bias=True)


class Resnext(nn.Module):
    def __init__(self, use_meta=False):
        super(Resnext, self).__init__()
        self.use_meta = use_meta
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=True)
        if use_meta:
            in_features = 2048 + 384
            self.meta = nn.Sequential(
                nn.Linear(2, 384),
                nn.ReLU(inplace=True),
                nn.LayerNorm(384),
                ResNormLayer(384),
            )
        else:
            in_features = 2048
        self.model.fc = nn.Linear(in_features=in_features, out_features=33, bias=True)


    def forward(self, input, meta=None):
        if self.use_meta:
            meta = meta[:, :2] # no time
            output_feature = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(input))))
            output_feature = self.model.layer4(self.model.layer3(self.model.layer2(self.model.layer1(output_feature))))
            output_feature = self.model.avgpool(output_feature)
            output_feature = torch.flatten(output_feature, 1)
            out_meta = self.meta(meta)
            agg_out = torch.cat([output_feature, out_meta], 1)
            output = self.model.fc(agg_out)
        else:
            output = self.model(input)
        return output


class Resnest(nn.Module):
    def __init__(self, use_meta=False):
        super(Resnest, self).__init__()
        self.use_meta = use_meta
        torch.hub.list('zhanghang1989/ResNeSt', force_reload=True)
        self.model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
        if use_meta:
            in_features = 2048 + 384
            self.meta = nn.Sequential(
                nn.Linear(2, 384),
                nn.ReLU(inplace=True),
                nn.LayerNorm(384),
                ResNormLayer(384),
            )
        else:
            in_features = 2048
        self.model.fc = nn.Linear(in_features=in_features, out_features=33, bias=True)

    def forward(self, input, meta=None):
        if self.use_meta:
            meta = meta[:, :2] # no time
            output_feature = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(input))))
            output_feature = self.model.layer4(self.model.layer3(self.model.layer2(self.model.layer1(output_feature))))
            output_feature = self.model.avgpool(output_feature)
            output_feature = torch.flatten(output_feature, 1)
            out_meta = self.meta(meta)
            agg_out = torch.cat([output_feature, out_meta], 1)
            output = self.model.fc(agg_out)
        else:
            output = self.model(input)
        return output