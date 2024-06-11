import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from .resnet_classifier import ResNetClassifier

import cv2
import os

def load(ckpt_path):
    model = ResNetClassifier.load_from_checkpoint(ckpt_path)

    return model


def preprocess(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (300,300))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img_rgb



def pred(img, model):
    prediction = model.predict(img)
    prob = prediction[0][0].item()
    prob = round(prob,2) * 100

    return prob
