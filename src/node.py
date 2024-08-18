import numpy as np


class Node:
    def __init__(self, data, trainable: bool = False):
        self.data = np.asarray(data)
        self.grad = np.zeros_like(data)
        self.trainable = trainable

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    def accumulate_grad(self, grad):
        if self.trainable:
            self.grad += grad

    def update_data(self, learning_rate=0.01):
        if self.trainable:
            self.data -= learning_rate * self.grad
            self.zero_grad()