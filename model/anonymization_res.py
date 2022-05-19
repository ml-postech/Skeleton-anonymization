from model.shift_gcn import Model as ActionClassifier
from model.shift_gcn import Model as PrivacyClassifier
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import sys
sys.path.insert(0, '')


class ResBlock(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(num_features, num_features)
        self.fc2 = nn.Linear(num_features, num_features)

    def forward(self, x):
        residual = x
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = x + residual
        out = x
        return out


class Anonymizer(nn.Module):
    def __init__(self, num_point, num_person, graph=None, graph_args=dict(), in_channels=3):
        super().__init__()

        num_features = num_person * num_point * in_channels

        self.res_blocks = nn.ModuleList(
            [ResBlock(num_features) for _ in range(10)])

    def forward(self, x):
        N, C, T, V, M = x.size()  # (batch, xyz, frames, joints, #people)

        # (batch, frames, xyz, joints, #people)
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(N, T, C * V * M)

        for block in self.res_blocks:
            x = block(x)

        # (batch, xyz, frames, joints, #people)
        out = x.view(N, T, C, V, M).permute(0, 2, 1, 3, 4)

        return out


class Model(nn.Module):
    def __init__(self,
                 num_privacy_class,
                 num_action_class,
                 num_point,
                 num_person,
                 graph,
                 graph_args,
                 in_channels=3):
        super().__init__()

        self.anonymizer = Anonymizer(
            num_point, num_person,
            graph, graph_args, in_channels)

    def load_action_classifier(self, path):
        self.action_classifier.load_state_dict(torch.load(path))

    def load_privacy_classifier(self, path):
        self.privacy_classifier.load_state_dict(torch.load(path))

    def forward(self, x):
        anonymized = self.anonymizer(x)

        privacy = self.privacy_classifier(anonymized)
        action = self.action_classifier(anonymized)

        return anonymized
