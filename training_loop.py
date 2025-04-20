import time
from train_settings import TrainSettings
from evaluator import Evaluator
import torch


class TrainingLoop:
    def __init__(self, settings: TrainSettings, evals: list[Evaluator] = []):
        self.current_epoch = 0
        self.ellapsed_time = 0
        self.tr_losses = []
        self.val_losses = []
        self.evaluations = []
        self.evaluators = evals
        self.settings = settings

    def train(self,  epochs=None):
        if epochs is None:
            epochs = self.settings.epochs
        if epochs > self.settings.epochs:
            return
        start_time = time.time()
        print(f"Training {self.settings.name} for {epochs} epochs")
        print(f"Training on {self.settings.device}")
        start_epoch = self.current_epoch
        for epoch in range(self.current_epoch, epochs):
            imgs, labels = imgs.to(self.settings.device), labels.to(
                self.settings.device)
            outputs = self.settings.model(imgs)
            if self.settings.print_memory:
                print(
                    f"Memory usage: {torch.cuda.memory_allocated(self.settings.device) / 1024**2:.2f} MB")
            loss = self.settings.loss_fn(outputs, labels)
            epoch_tr_loss += loss.item()

            self.settings.optimizer.zero_grad()
            loss.backward()
            self.settings.optimizer.step()
            if (self.settings.print_steps):
                eta = time.time() - start_time
                print(f"Step {step}, Ellapsed {eta:.2f} seconds, Train Loss: {loss.item():.4f}, ETA: {TrainingLoop.calculate_step_eta(epoch-start_epoch, epochs - start_epoch, step, len(self.settings.train_data), len(self.settings.val_data), eta):.2f} seconds")
            step += 1

            evaluations = []
            self.settings.model.eval()
            with torch.no_grad():
                for imgs, labels in self.settings.val_data:
                    imgs, labels = imgs.to(self.settings.device), labels.to(
                        self.settings.device)
                    outputs = self.settings.model(imgs)
                    loss = self.settings.loss_fn(outputs, labels)
                    epoch_val_loss += loss.item()
                if self.current_epoch % self.settings.eval_after_epoch == 0:
                    for eval in self.evaluators:
                        evaluation = eval(
                            self.settings.model, self.settings.val_data, self.settings.device)
                        evaluations.append((eval.name(), evaluation))

            epoch_tr_loss /= len(self.settings.train_data)
            epoch_val_loss /= len(self.settings.val_data)
            self.tr_losses.append(epoch_tr_loss)
            self.val_losses.append(epoch_val_loss)
            self.evaluations.append(evaluations)
            self.settings.save_if_needed(epoch)

            if (epoch + 1) % self.settings.eval_after_epoch == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Ellapsed {(time.time() - start_time):.2f} seconds, Train Loss: {epoch_tr_loss:.4f}, Validation Loss: {epoch_val_loss:.4f} Evaluations: {evaluations} ETA: {self.ellapsed_time / (epoch + 1) * (epochs - epoch - 1):.2f} seconds")

        self.settings.save_final()
        end_time = time.time() - start_time
        self.ellapsed_time += end_time

    @staticmethod
    def calculate_step_eta(current_epoch, epochs, current_step, tr_len,  val_len,  ellapsed_time):
        step_per_epoch = tr_len + val_len
        epochs_left = epochs - current_epoch
        steps_left = step_per_epoch * epochs_left - current_step - 1
        steps_already_taken = current_epoch * step_per_epoch + current_step + 1
        step_time = ellapsed_time / steps_already_taken
        eta = step_time * steps_left
        return eta
