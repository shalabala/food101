import torch

class Evaluator:
    def __call__(self, model, dataloader, device):
        with torch.no_grad():
            self.eval(model, dataloader, device)
    
    def name(self):
        return self.__class__.__name__