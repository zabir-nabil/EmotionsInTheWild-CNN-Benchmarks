"""
cnn model implementation
"""
import torch
from torchvision import models
from torch import nn
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import Conv2dStaticSamePadding
from config import Config as cfg

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from dataloader import EmotionDataset
from tqdm import tqdm
from sklearn.metrics import f1_score


class EmotionCNN(nn.Module):
    def __init__(self, cnn):
        super(EmotionCNN, self).__init__()

        self.cnn_name = cnn
        
        # CNN pre-trained models
        cnn_dict = {'resnet18': models.resnet18, 'resnet34': models.resnet34, 'resnet50': models.resnet50,
                   'resnet101': models.resnet101, 'resnet152': models.resnet152, 'resnext50': models.resnext50_32x4d,
                   'resnext101': models.resnext101_32x8d}
        
        # feature dim
        self.out_dict = {'resnet18': 512, 'resnet34': 512, 'resnet50': 2048, 'resnet101': 2048, 'resnet152': 2048,
                         'resnext50': 2048, 'resnext101': 2048, "efnb0": 1280, "efnb1": 1280, "efnb2": 1408, 
                          "efnb3": 1536, "efnb4": 1792, "efnb5": 2048, "efnb6": 2304, "efnb7": 2560}
        
        
        # efficient net b0 to b7
        
        if cnn in cnn_dict.keys(): # resnet or resnext
            self.emo_cnn = cnn_dict[cnn](pretrained = True)
            
            # make arbitrary n channel(s)
            if cfg.n_channels != 3:
                self.emo_cnn.conv1 = nn.Conv2d(cfg.n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            
            # remove the fc layer/ add a simple linear layer
            self.emo_cnn.fc = nn.Linear(self.out_dict[cnn], cfg.n_classes)   # no intemediate linear
            
        elif cnn == "efnb0":
            self.emo_cnn = EfficientNet.from_pretrained('efficientnet-b0')
            if cfg.n_channels != 3:
                self.emo_cnn._conv_stem = Conv2dStaticSamePadding(cfg.n_channels, 32, kernel_size = (3,3), stride = (2,2), 
                                                                bias = False, image_size = 512)
            self.emo_cnn._fc = nn.Linear(self.out_dict[cnn], cfg.n_classes)
            self.emo_cnn._swish = nn.Identity()
        
        elif cnn == "efnb1":
            self.emo_cnn = EfficientNet.from_pretrained('efficientnet-b1')
            if cfg.n_channels != 3:
                self.emo_cnn._conv_stem = Conv2dStaticSamePadding(cfg.n_channels, 32, kernel_size = (3,3), stride = (2,2), 
                                                                bias = False, image_size = 512)
            self.emo_cnn._fc = nn.Linear(self.out_dict[cnn], cfg.n_classes)
            self.emo_cnn._swish = nn.Identity()

        elif cnn == "efnb2":
            self.emo_cnn = EfficientNet.from_pretrained('efficientnet-b2')
            if cfg.n_channels != 3:
                self.emo_cnn._conv_stem = Conv2dStaticSamePadding(cfg.n_channels, 32, kernel_size = (3,3), stride = (2,2), 
                                                                bias = False, image_size = 512)
            self.emo_cnn._fc = nn.Linear(self.out_dict[cnn], cfg.n_classes)
            self.emo_cnn._swish = nn.Identity()
        
        elif cnn == "efnb3":
            self.emo_cnn = EfficientNet.from_pretrained('efficientnet-b3')
            if cfg.n_channels != 3:
                self.emo_cnn._conv_stem = Conv2dStaticSamePadding(cfg.n_channels, 40, kernel_size = (3,3), stride = (2,2), 
                                                                bias = False, image_size = 512)
            self.emo_cnn._fc = nn.Linear(self.out_dict[cnn], cfg.n_classes)
            self.emo_cnn._swish = nn.Identity()
        
        elif cnn == "efnb4":
            self.emo_cnn = EfficientNet.from_pretrained('efficientnet-b4')
            if cfg.n_channels != 3:
                self.emo_cnn._conv_stem = Conv2dStaticSamePadding(cfg.n_channels, 48, kernel_size = (3,3), stride = (2,2), 
                                                                bias = False, image_size = 512)
            self.emo_cnn._fc = nn.Linear(self.out_dict[cnn], cfg.n_classes)
            self.emo_cnn._swish = nn.Identity()
        
        elif cnn == "efnb5":
            self.emo_cnn = EfficientNet.from_pretrained('efficientnet-b5')
            if cfg.n_channels != 3:
                self.emo_cnn._conv_stem = Conv2dStaticSamePadding(cfg.n_channels, 48, kernel_size = (3,3), stride = (2,2), 
                                                                bias = False, image_size = 512)
            self.emo_cnn._fc = nn.Linear(self.out_dict[cnn], cfg.n_classes)
            self.emo_cnn._swish = nn.Identity()
        
        elif cnn == "efnb6":
            self.emo_cnn = EfficientNet.from_pretrained('efficientnet-b6')
            if cfg.n_channels != 3:
                self.emo_cnn._conv_stem = Conv2dStaticSamePadding(cfg.n_channels, 56, kernel_size = (3,3), stride = (2,2), 
                                                                bias = False, image_size = 512)
            self.emo_cnn._fc = nn.Linear(self.out_dict[cnn], cfg.n_classes)
            self.emo_cnn._swish = nn.Identity()
        
        elif cnn == "efnb7":
            self.emo_cnn = EfficientNet.from_pretrained('efficientnet-b7')
            if cfg.n_channels != 3:
                self.emo_cnn._conv_stem = Conv2dStaticSamePadding(cfg.n_channels, 64, kernel_size = (3,3), stride = (2,2), 
                                                                bias = False, image_size = 512)
            self.emo_cnn._fc = nn.Linear(self.out_dict[cnn], cfg.n_classes)
            self.emo_cnn._swish = nn.Identity()
        
        else:
            raise ValueError("cnn not recognized")
        
            
    def forward(self, x):
        x = self.emo_cnn(x) # out
        
        return x

class EmotionTrainer:
    def __init__(self, train_loader = None, val_loader = None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(f"cuda:{cfg.gpu_index}" if torch.cuda.is_available() else "cpu")
        
        # loss
        if cfg.classification == "multi-label":
            self.criterion = nn.BCEWithLogitsLoss()
        
        # build train, val dataloader
        train_dataset = EmotionDataset(cfg.dataset, split="train")
        val_dataset = EmotionDataset(cfg.dataset, split="val")
        self.data_loader = {}
        self.data_loader['train'] = DataLoader(train_dataset, batch_size = cfg.batch_size, shuffle = True, num_workers = cfg.n_workers)
        self.data_loader['val'] = DataLoader(val_dataset, batch_size = cfg.batch_size, shuffle = True, num_workers = cfg.n_workers)
        

    def train(self, model, n_epochs = 1):
        print(f"Training {model.cnn_name} for {n_epochs} epochs.")
        self.model = model
        # take model to device
        self.model = model.to(self.device)
        # optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience = 5)

        for epoch in range(n_epochs):
            print(f"Epoch {epoch+1}")
            result = []
            for phase in ['train', 'val']:
                if phase == "train":  # training
                    self.model.train()
                else:  # validation
                    self.model.eval()
            
                # keep track of training and validation loss
                running_loss = 0.0
                running_corrects = 0.0  
            
                for x, y in tqdm(self.data_loader[phase]):
                    # load the data and target to respective device
                    x, y = x.to(self.device)  , y.to(self.device)

                    with torch.set_grad_enabled(phase=="train"):
                        output = self.model(x)
                        # calculate the loss
                        loss = self.criterion(output,y)
                        preds = torch.sigmoid(output).data > 0.5
                        preds = preds.to(torch.float32)
                        
                        if phase=="train"  :
                            # backward pass: compute gradient of the loss with respect to model parameters 
                            loss.backward()
                            # update the model parameters
                            self.optimizer.step()
                            # zero the grad to stop it from accumulating
                            self.optimizer.zero_grad()


                    # statistics
                    running_loss += loss.item() * x.size(0)
                    running_corrects += f1_score(y.to("cpu").to(torch.int).numpy(), preds.to("cpu").to(torch.int).numpy() , average="samples")  * x.size(0)
                    # average = samples, is only for multi-label
                    
                    
                epoch_loss = running_loss / len(data_loader[phase])
                epoch_acc = running_corrects / len(data_loader[phase])

                if phase == "val":
                    self.scheduler.step(epoch_acc)


                result.append('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print(result)


