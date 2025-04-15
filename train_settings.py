import torch
import os
from torch.utils.data import random_split, DataLoader


class TrainSettings:
    def __init__(self,
                 name,
                 model,
                 dataset_tr,
                 dataset_val,
                 epochs=100,
                 eval_after_epoch=10,
                 save_path=None,
                 device='cpu',
                 batch_size=64,
                 optimizer_type='adam',
                 lr=0e-3,
                 momentum=0.9,
                 save_after_epoch=None,
                 print_steps = False):

        if save_after_epoch is None:
            save_after_epoch = eval_after_epoch

        if save_path is None:
            save_path = os.path.join(os.getcwd(), f"{model.name()}_saves")
            if not os.path.exists(save_path):
                os.makedirs(save_path)

        self.model = model
        self.lr = lr,
        self.device = device

        self.epochs = epochs
        self.model.to(device)

        self.batch_size = batch_size
        self.optimizer_type = optimizer_type
        self.momentum = momentum
        self.optimizer_type = optimizer_type

        self.eval_after_epoch = eval_after_epoch

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.train_data = DataLoader(
            dataset_tr, batch_size=batch_size, shuffle=True)
        self.val_data = DataLoader(
            dataset_val, batch_size=batch_size, shuffle=False)
        self.save_path = save_path
        self.save_after_epoch = save_after_epoch
        self.name = name

        if optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer_type == 'sgd':
            self.optimizer = torch.optim.SGD(
                model.parameters(), lr=lr, momentum=momentum)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        self.print_steps = print_steps

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
            torch.save(self.model.state_dict(), os.path.join(
                self.save_path, f'{self.name}_epoch_{epoch}.pth'))

    def save_final(self):
        torch.save(self.model.state_dict(), os.path.join(
            self.save_path, f'{self.name}_final.pth'))
