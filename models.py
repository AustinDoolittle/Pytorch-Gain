import torch
import torchvision
import sys

stock_models = [i for i in dir(torchvision.models) if not i.startswith('_')]
custom_models = ['GreyNet19', 'Darknet53']
available_models = stock_models + custom_models

def get_model(model_type, output_length, batch_norm, num_channels=3):
    if model_type in custom_models:
        # hacky, but worky
        model = globals()[model_type]
        model = model(output_length, num_channels, batch_norm=batch_norm)
    else:
        model = getattr(torchvision.models, model_type)
        model = model(pretrained=False, num_classes=output_length)
    return model

def model_to_str(model):
    ret_str = ''
    for idx, m in model.named_modules():
        ret_str += '%s: %s\n'%(str(idx), str(m))
    return ret_str

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_planes, batch_norm=True, **kwargs):
        # roughly based off of https://github.com/pytorch/vision/blob/1a6038eaa5aeafbb46c78a5d57d9a42e3d90f1f7/torchvision/models/resnet.py#L57
        super(ResidualBlock, self).__init__(**kwargs)
        seq_list = []
        seq_list.append(torch.nn.Conv2d(in_planes, in_planes / 2, 1, padding=1))
        if batch_norm:
            seq_list.append(torch.nn.BatchNorm2d(in_planes / 2))

        seq_list.append(torch.nn.Conv2d(in_planes / 2, in_planes, 3))
        if batch_norm:
            seq_list.append(torch.nn.BatchNorm2d(in_planes))
        self.seq = torch.nn.Sequential(*seq_list)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out = self.seq(x)
        out = out + x
        out = self.relu(out)

        return out

class Darknet53(torch.nn.Module):
    def __init__(self, class_count, channel_count, batch_norm=True, **kwargs):
        super(Darknet53, self).__init__(**kwargs)
        seq_list = []
        seq_list.append(torch.nn.Conv2d(channel_count, 32, 3, padding=1))
        if batch_norm:
            seq_list.append(torch.nn.BatchNorm2d(32))
        seq_list.append(torch.nn.Conv2d(32, 64, 3, stride=2, padding=1))
        if batch_norm:
            seq_list.append(torch.nn.BatchNorm2d(64))

        seq_list.append(ResidualBlock(64))
        seq_list.append(torch.nn.Conv2d(64, 128, 3, stride=2, padding=1))
        if batch_norm:
            seq_list.append(torch.nn.BatchNorm2d(128))

        for i in range(2):
            seq_list.append(ResidualBlock(128, batch_norm=batch_norm))

        seq_list.append(torch.nn.Conv2d(128, 256, 3, stride=2, padding=1))
        if batch_norm:
            seq_list.append(torch.nn.BatchNorm2d(256))

        for i in range(8):
            seq_list.append(ResidualBlock(256, batch_norm=batch_norm))

        seq_list.append(torch.nn.Conv2d(256, 512, 3, stride=2, padding=1))
        if batch_norm:
            seq_list.append(torch.nn.BatchNorm2d(512))

        for i in range(8):
            seq_list.append(ResidualBlock(512, batch_norm=batch_norm))

        seq_list.append(torch.nn.Conv2d(512, 512, 3, padding=1))
        # if batch_norm:
            # seq_list.append(torch.nn.BatchNorm2d(1024))

        # for i in range(4):
            # seq_list.append(ResidualBlock(1024, batch_norm=batch_norm))

        # seq_list.append(torch.nn.Conv2d(1024, 1024, 3))
        seq_list.append(torch.nn.ReLU())
        self.seq = torch.nn.Sequential(*seq_list)

        classifier_list = [
            # torch.nn.Linear(1024 * 8 * 8, 1024),
            torch.nn.Linear(512 * 16 * 16, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1024, class_count),
        ]

        self.fc = torch.nn.Sequential(*classifier_list)


    def forward(self, x):
        out = self.seq(x)
        out = self.fc(out.view(out.size()[0], -1))

        return out

class GreyNet19(torch.nn.Module):
    # custom simple network to use for heatmap generation
    # based off of darknet-53 defined here: https://pjreddie.com/media/files/papers/YOLOv3.pdf
    # just shortened the network and increased size of output feature map
    def __init__(self, class_count, channel_count, batch_norm=True, **kwargs):
        super(GreyNet19, self).__init__(**kwargs)
        seq_list = []
        seq_list.append(torch.nn.Conv2d(channel_count, 32, 3, padding=1))
        if batch_norm:
            seq_list.append(torch.nn.BatchNorm2d(32))
        seq_list.append(torch.nn.Conv2d(32, 64, 3, stride=2, padding=1))
        if batch_norm:
            seq_list.append(torch.nn.BatchNorm2d(64))

        seq_list.append(ResidualBlock(64, batch_norm=batch_norm))
        seq_list.append(torch.nn.Conv2d(64, 128, 3, stride=2, padding=1))
        if batch_norm:
            seq_list.append(torch.nn.BatchNorm2d(128))

        for i in range(2):
            seq_list.append(ResidualBlock(128, batch_norm=batch_norm))

        seq_list.append(torch.nn.Conv2d(128, 256, 3, stride=2, padding=1))
        if batch_norm:
            seq_list.append(torch.nn.BatchNorm2d(256))

        for i in range(4):
            seq_list.append(ResidualBlock(256))

        seq_list.append(torch.nn.Conv2d(256, 512, 3, stride=2, padding=1))

        seq_list.append(torch.nn.ReLU())
        # seq_list.append(torch.nn.AdaptiveAvgPool2d(1))
        self.seq = torch.nn.Sequential(*seq_list)

        classifier_list = [
            torch.nn.Linear(512 * 16 * 16, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1024, class_count),
        ]
        self.classifier = torch.nn.Sequential(*classifier_list)

    def forward(self, x):
        out = self.seq(x)
        return self.classifier(out.view(out.size()[0], -1))












