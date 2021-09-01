"""
dataloader for training on emotic data
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import json
import os, random
from config import Config as cfg

class EmotionDataset(Dataset):
    def __init__(self, dataset = "emotic", split = "train"):
        self.images = [] # path to images
        self.labels = [] # label for each image
        self.bbox = [] # this is optional, but helps training better models for emotic
        if dataset == "emotic":
            annotations = json.load(open(cfg.annotations_path))
            self.classes = annotations["emotions"]
            for img in annotations[split]:
                if os.path.exists(os.path.join(cfg.dataset_path, img['path'])):
                    self.images.append(os.path.join(cfg.dataset_path, img['path']))
                    self.labels.append([self.classes.index(emo) for emo in img['labels']])
                    self.bbox.append(img['bbox'])
        # image pipeline
        self.compose = []
        self.compose.append(transforms.Resize(cfg.image_size, Image.BICUBIC))
        try:
            if cfg.augment:
                self.compose.extend(
                    [transforms.RandomPerspective(distortion_scale=0.6, p=0.5),
                    transforms.RandomRotation(degrees=(0, 180)),
                    transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)),
                    transforms.RandomInvert(p=0.2),
                    transforms.RandomAdjustSharpness(0, p=0.25),
                    transforms.RandomAdjustSharpness(2, p=0.25),
                    transforms.RandomAutocontrast(p=0.2),
                    transforms.RandomEqualize(p=0.1)])
                print("Augmentation will be applied")
        except:
            print("Augmentation can not be applied. You have a lower pytorch version.")
        self.compose.append(transforms.ToTensor())
        self.transform = transforms.Compose(self.compose)
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        x = Image.open(self.images[idx]).convert('RGB')
        if len(self.bbox) != 0 and random.randint(0,1) == 1: # bbox
            x = x.crop(self.bbox[idx])
        

        # apply transformation
        x = self.transform(x)
        # simple normalization
        if cfg.normalize_images:
            pass
            # placeholder normalization

        y = torch.zeros(cfg.n_classes)
        for y_i in self.labels[idx]:
            y[y_i] = 1
        return x, y

if __name__ == "__main__":
    # test dataloader
    dataset = EmotionDataset(split="val")
    dataloader = DataLoader(dataset, batch_size = 4, shuffle = True, num_workers = 2)
    for x, y in dataloader:
        print(x.shape)
        print(y.shape)
        print("Stats:")
        print("-------------")
        print(x.mean())
        print(x.min())
        print(x.max())
        break