"""
training + testing config
author: github.com/zabir-nabil
"""
class Config:
    dataset = "emotic"
    dataset_path = "data/emotic"
    annotations_path = "data/annotations.json"
    n_classes = 26
    n_channels = 3 # how many channels are there in the input image/data
    image_size = [512, 512]
    augment = True
    normalize_images = False
    classification = "multi-label" # multi-label, multi-class
    gpu_index = 0
    multi_gpus = [0, 1] # if there is more than one gpu, you can pass a list [0, 1], else just use the gpu index (0 or 1 or 2, etc.)
    batch_size = 32
    n_workers = 8 # set 0 for windows
    models = ['resnet18', 'resnet34', 'resnet50', 'resnext50', 'efnb0', 'efnb1', 'efnb2', 'efnb3', 'efnb4', 'efnb5'] # see the other model options in model.py
    n_epochs = 50
 
    