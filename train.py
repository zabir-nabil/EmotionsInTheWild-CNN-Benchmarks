"""
training code
author: github.com/zabir-nabil
"""
from config import Config as cfg
from dataloader import EmotionDataset
from model import EmotionCNN, EmotionTrainer
import torch 

print("hyperparameters:")
print(dict(cfg.__dict__))

model_trainer = EmotionTrainer()

for model_name in cfg.models:
    model = EmotionCNN(model_name)
    # check if multi-gpu training is enabled
    if torch.cuda.device_count() > 1 and type(cfg.multi_gpus) == list:
        model = torch.nn.DataParallel(model, device_ids=cfg.multi_gpus)
    model_trainer.train(model, n_epochs=cfg.n_epochs)
    
