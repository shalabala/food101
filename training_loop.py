import time
from train_settings import TrainSettings
from evaluator import Evaluator
import torch

class TrainingLoop:
    def __init__(self, settings : TrainSettings, evals : list[Evaluator] = []):
        self.current_epoch = 0
        self.ellapsed_time = 0
        self.tr_losses = []
        self.val_losses = []
        self.evaluations = []
        self.evaluators = evals
        self.settings = settings
    
    def train(self,  epochs = None):
        if epochs is None:
            epochs = self.settings.epochs
        if epochs > self.settings.epochs:
            return
        start_time = time.time()
        
        for epoch in range(self.current_epoch, epochs):
            epoch_tr_loss = 0.0
            epoch_val_loss = 0.0
            self.current_epoch = epoch
            for imgs, labels in self.settings.train_data:
                imgs, labels = imgs.to(self.settings.device), labels.to(self.settings.device)
                outputs = self.settings.model(imgs)
                loss = self.settings.loss_fn(outputs, labels)
                epoch_tr_loss += loss.item()
                
                self.settings.optimizer.zero_grad()
                loss.backward()
                self.settings.optimizer.step()
            
            evaluations = []
            with torch.no_grad():
                for imgs, labels in self.settings.val_data:
                    imgs, labels = imgs.to(self.settings.device), labels.to(self.settings.device)
                    outputs = self.settings.model(imgs)
                    loss = self.settings.loss_fn(outputs, labels)
                    epoch_val_loss += loss.item()
                if self.current_epoch % self.settings.eval_after_epoch == 0:
                    for eval in self.evaluators:
                        evaluation = eval(self.settings.model, self.settings.val_data, self.settings.device)
                        evaluations.append((eval.name(), evaluation))
                        
            epoch_tr_loss /= len(self.settings.train_data)
            epoch_val_loss /= len(self.settings.val_data)
            self.tr_losses.append(epoch_tr_loss)
            self.val_losses.append(epoch_val_loss)
            self.evaluations.append(evaluations)
            self.settings.save_if_needed(epoch)

            if(epoch + 1) % self.settings.eval_after_epoch == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_tr_loss:.4f}, Validation Loss: {epoch_val_loss:.4f} Evaluations: {evaluations}") 
         
        self.settings.save_final()       
        end_time = time.time() - start_time
        self.ellapsed_time += end_time    