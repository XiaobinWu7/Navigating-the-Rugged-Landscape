import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from models.pointnet_utils import PointNetEncoder, feature_transform_reguliarzer
from torch.autograd import Variable


class get_model(nn.Module):
    def __init__(self, k=40, normal_channel=True):
        super(get_model, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x


class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss


class softmax_with_smoothing_label_loss(nn.Module):
    def __init__(self, num_classes=40, label_smoothing=0.2):
        super(softmax_with_smoothing_label_loss, self).__init__()
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.ones = None

    def forward(self, output, target):
        if self.ones is None:
            self.ones = Variable(torch.eye(self.num_classes).to(output).cuda())
        output = -1*self.log_softmax(output)
        one_hot = self.ones.index_select(0,target)
        one_hot = one_hot*(1 - self.label_smoothing) + self.label_smoothing / self.num_classes

        loss = one_hot * output
        loss = loss.sum(dim=1)
        loss = loss.mean()

        return loss