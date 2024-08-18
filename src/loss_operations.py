import numpy as np
from operation import Operation
from node import Node

class MeanSquaredError(Operation):
    def forward(self, y_pred: Node, y_true: Node) -> Node:
        assert y_pred.data.shape == y_true.data.shape

        self.y_pred = y_pred
        self.y_true = y_true
        value = np.mean((y_pred.data - y_true.data) ** 2)
        return Node(value)
    
    def backward(self, upstream_grad):
        sz = self.y_true.data.size
        dy_pred = (2 / sz) * (self.y_pred.data - self.y_true.data) * upstream_grad
        dy_true = (-2 / sz) * (self.y_pred.data - self.y_true.data) * upstream_grad

        self.y_pred.accumulate_grad(dy_pred)
        self.y_true.accumulate_grad(dy_true)
        return dy_pred, dy_true
    
class MeanAbsoluteError(Operation):
    def forward(self, y_pred: Node, y_true: Node) -> Node:
        assert y_pred.data.shape == y_true.data.shape

        self.y_pred = y_pred
        self.y_true = y_true
        value = np.mean(np.abs(y_pred.data - y_true.data))
        return Node(value)
    
    def backward(self, upstream_grad):
        sz = self.y_true.data.size
        dy_pred = (1 / sz) * np.sign(self.y_pred.data - self.y_true.data) * upstream_grad
        dy_true = (-1 / sz) * np.sign(self.y_pred.data - self.y_true.data) * upstream_grad

        self.y_pred.accumulate_grad(dy_pred)
        self.y_true.accumulate_grad(dy_true)
        return dy_pred, dy_true
    
class CrossEntropyLoss(Operation):
    def forward(self, y_pred: Node, y_true: Node) -> Node:
        assert y_pred.data.shape == y_true.data.shape

        self.y_pred = y_pred
        self.y_true = y_true

        eps = 1e-15
        value = -np.mean(y_true.data * np.log(y_pred.data + 1e-15))
        return Node(value)
    
    def backward(self, upstream_grad):
        sz = self.y_true.data.size
        dy_pred = -(self.y_true.data / (self.y_pred.data + 1e-15)) * upstream_grad / sz
        dy_true = (-np.log(self.y_pred.data + 1e-15)) * upstream_grad / sz

        self.y_pred.accumulate_grad(dy_pred)
        self.y_true.accumulate_grad(dy_true)
        return dy_pred, dy_true
    
