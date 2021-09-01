"""
training code
author: github.com/zabir-nabil
"""
from config import Config as cfg
from dataloader import EmotionDataset
from model import EmotionCNN, EmotionTrainer

print("hyperparameters:")
print(dict(cfg.__dict__))

model_trainer = EmotionTrainer()

for model_name in cfg.models:
    model = EmotionCNN(model_name)
    model_trainer.train(model, n_epochs=cfg.n_epochs)
    
