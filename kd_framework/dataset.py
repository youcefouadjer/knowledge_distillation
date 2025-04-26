import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import LFWPeople, CelebA
from PIL import Image
import os

# ---------------------- Data Preparation ----------------------

class LFWTransformDataset(Dataset):
    def __init__(self, split, transform_teacher, transform_student):
        self.base = LFWPeople(root='.', split=split, download=True)
        self.transform_teacher = transform_teacher
        self.transform_student = transform_student

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        img_teacher = self.transform_teacher(img)
        img_student = self.transform_student(img)
        return img_teacher, img_student, label


class celebATransformDataset(Dataset):
    def __init__(self, split, transform_teacher, transform_student, id2label):
        self.base = CelebA(
            root=".",
            split=split,
            target_type='identity',
            download=False
        )
        self.transform_teacher = transform_teacher
        self.transform_student = transform_student
        self.id2label = id2label

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, identity = self.base[idx]
        img_teacher = self.transform_teacher(img)
        img_student = self.transform_student(img)
        label = self.id2label[int(identity)]  # remapping

        return img_teacher, img_student, label

