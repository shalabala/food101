import time
from train_logic import TrainLogic
from train_settings import TrainSettings
from evaluator import Evaluator
import torch
import utility


class TrainingLoop:
    def __init__(self, settings: TrainSettings, train_logic: TrainLogic, evals: list[Evaluator] = []):
        self.current_epoch = -1
        self.ellapsed_time = 0
        self.tr_losses = []
        self.val_losses = []
        self.evaluations = []
        self.evaluators = evals
        self.settings = settings
        self.train_logic = train_logic

    def train(self,  epochs=None):
        if epochs is None:
            epochs = self.settings.epochs
        if epochs < self.current_epoch:
            return
        start_time = time.time()
        print(f"Training {self.settings.name} for {epochs} epochs")
        print(f"Training on {self.settings.device}")
        start_epoch = self.current_epoch
        for epoch in range(self.current_epoch + 1, epochs):
            self.settings.model.train()
            epoch_tr_loss = 0.0
            epoch_val_loss = 0.0
            self.current_epoch = epoch
            step = 0
            for imgs, labels in self.settings.train_data:
                loss = self.train_logic.train(
                    self.settings.device,
                    imgs,
                    labels,
                    self.settings.model,
                    self.settings.optimizer,
                    self.settings.loss_fn
                )
                epoch_tr_loss += loss

                if self.settings.print_after_steps > 0 and (step + 1) % self.settings.print_after_steps == 0:
                    if self.settings.print_memory:
                        print(
                            f"Memory usage: {torch.cuda.memory_allocated(self.settings.device) / 1024**2:.2f} MB")
                    eta = time.time() - start_time
                    ellapsed_time_string = utility.time_string(eta)
                    eta_time_string = utility.time_string(TrainingLoop.calculate_step_eta(
                        epoch-start_epoch-1,
                        epochs-start_epoch-1,
                        step,
                        len(self.settings.train_data),
                        len(self.settings.val_data), eta))
                    print(
                        f"[{utility.current_time_for_log()}] Epoch {epoch+1} Step {step+1}/{len(self.settings.train_data)}, Ellapsed {ellapsed_time_string}, Train Loss: {loss:.4f}, ETA: {eta_time_string}")
                step += 1

            evaluations = []
            self.settings.model.eval()
            with torch.no_grad():
                epoch_val_loss, evaluations = self.train_logic.validate(
                    self.settings.val_data,
                    self.settings.device,
                    self.settings.model,
                    self.settings.loss_fn,
                    self.evaluators
                )

            normalized_tr_loss = TrainingLoop.normalize_loss(
                epoch_tr_loss, len(self.settings.train_data))
            normalized_val_loss = TrainingLoop.normalize_loss(
                epoch_val_loss, len(self.settings.val_data))

            self.tr_losses.append(normalized_tr_loss)
            self.val_losses.append(normalized_val_loss)
            self.evaluations.append(evaluations)

            # if this is final epoch it would be saved anyways by save_final
            if epoch + 1 != epochs:
                self.settings.save_if_needed(epoch+1)

            if (epoch + 1) % self.settings.eval_after_epoch == 0:
                ellapsed_time = time.time() - start_time
                eta_time_string = utility.time_string(
                    ellapsed_time / (epoch + 1) * (epochs - epoch - 1))
                ellapsed_time_string = utility.time_string(ellapsed_time)
                print(f"[{utility.current_time_for_log()}] Epoch {epoch + 1}/{epochs}, Ellapsed {ellapsed_time_string} seconds, " +
                      f"Train Loss: {normalized_tr_loss:.4f}, Validation Loss: {normalized_val_loss:.4f} Evaluations: " +
                      f"{evaluations} ETA: {eta_time_string}")

        self.settings.save_final(epoch)
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

    @staticmethod
    def normalize_loss(loss: float, steps: int = -1) -> float:
        return loss / steps if steps > 0 else loss