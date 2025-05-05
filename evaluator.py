import torch
from abc import ABC, abstractmethod


class Evaluator(ABC):
    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def pass_predictions(self, predictions, labels):
        pass

    @abstractmethod
    def get_value(self):
        pass

    def __str__(self):
        return self.__class__.__name__


class AccuracyEvaluator(Evaluator):
    def __init__(self):
        self.correct = 0
        self.total = 0

    def clear(self):
        self.correct = 0
        self.total = 0

    def pass_predictions(self, predictions, labels):
        _, predicted = torch.max(predictions.data, 1)
        self.total += labels.size(0)
        self.correct += (predicted == labels).sum().item()

    def get_value(self):
        if self.total == 0:
            return 0.0
        return self.correct / self.total

    def __str__(self):
        return "ACC"
