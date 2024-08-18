from node import Node
from basic_operations import Add, Multiply

import numpy as np


class DenseLayer:
    def __init__(self, input_size, output_size, random_or_zeros='random', trainable=True):
        super().__init__()
        
        # Initialize weights and biases; Biases are initialized to zeros, weights are initialized randomly or to zeros as specified
        if random_or_zeros == 'random':
            self.weights = Node(np.random.randn(input_size, output_size), trainable=trainable)
        elif random_or_zeros == 'zeros':
            self.weights = Node(np.zeros((input_size, output_size)), trainable=trainable)
        else:
            raise ValueError('random_or_zeros must be either "random" or "zeros"')
        
        self.biases = Node(np.zeros((1, output_size)), trainable=trainable)

        self.multiply_op = Multiply()
        self.add_op = Add()

        self.output = None

    
    def forward(self, input_node: Node):
        """Compute the forward pass through the layer"""

        linear_output = self.multiply_op.forward(input_node, self.weights)
        self.output = self.add_op.forward(linear_output, self.biases)

        return self.output

    
    def backward(self, upstream_grad):
        """Compute the backward pass through the layer"""
        grad_biases, grad_linear_output = self.add_op.backward(upstream_grad)
        input_grad, grad_weights = self.multiply_op.backward(grad_linear_output)

        return input_grad, grad_weights, grad_biases
    
    def update_parameters(self, learning_rate):
        """Update the weights and biases of the layer"""
        self.weights.update_data(learning_rate)
        self.biases.update_data(learning_rate)
        