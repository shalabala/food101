import torch
import os
from torch.utils.data import random_split, DataLoader

class TrainSettings:
    def __init__(self,
                 name,
                 model,
                 device, 
                 dataset_tr,
                 dataset_val,
                 batch_size,
                 optimizer_type,
                 lr, 
                 momentum,
                 save_path, 
                 save_after_epoch,
                 eval_after_epoch):
        
        self.model = model
        self.lr = lr,
        self.device = device
        
        self.model.to(device)
        
        self.batch_size = batch_size
        self.optimizer_type = optimizer_type
        self.momentum = momentum
        self.optimizer_type = optimizer_type

        self.eval_after_epoch = eval_after_epoch
        
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.train_data = DataLoader(dataset_tr, batch_size=batch_size, shuffle=True)
        self.val_data = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
        self.save_path = save_path
        self.save_after_epoch = save_after_epoch
        self.name = name
        
        if optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer_type == 'sgd':
            self.optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
    def properties(self):
        return {
            'model': self.model.name(),
            'optimizer': self.optimizer_type,
            'loss_fn': "CrossEntropyLoss",
            'save_path': self.save_path,
            'batch_size': self.batch_size,
            'learning_rate': self.lr,
            'momentum': self.momentum,
            'save_after_epoch': self.save_after_epoch,
            'name': self.name
        }
        
    def save_if_needed(self, epoch):
        if epoch % self.save_after_epoch == 0:
            torch.save(self.model.state_dict(), os.path.join(self.save_path, f'{self.name}_epoch_{epoch}.pth'))
            
    def save_final(self):
        torch.save(self.model.state_dict(), os.path.join(self.save_path, f'{self.name}_final.pth'))