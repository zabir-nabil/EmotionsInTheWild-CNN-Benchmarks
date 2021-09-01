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
    gpu_index = 0 # if there is more than one gpu
    batch_size = 4
    n_workers = 0 # set 0 for windows
    models = ["resnet18", "efnb0"] # see the other model options in model.py
    n_epochs = 1
 
    