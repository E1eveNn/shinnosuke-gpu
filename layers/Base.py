from .Core import Layer
import cupy as cp
from typing import Tuple




class Input(Layer):
    def __init__(self, shape: Tuple, value:cp.ndarray = None, **kwargs):
        super(Input,self).__init__(**kwargs)
        self.input_shape = shape
        self.output_shape = self.input_shape
        self.output_tensor = value
        self.require_grads = False



    def connect(self, prev_layer: Layer):
        if prev_layer is not None:
            self.input_shape = prev_layer.output_shape
            self.output_shape = self.input_shape




    def forward(self, is_training: bool):
        self.output_tensor = self.input_tensor
        super(Input, self).forward(is_training)


    def backward(self):
        pass


class Reshape(Layer):
    def __init__(self, output_shape: Tuple, **kwargs):
        super(Reshape,self).__init__(**kwargs)
        self.output_shape = output_shape



    def connect(self, prev_layer: Layer):
        if prev_layer is not None:
            self.input_shape = prev_layer.output_shape
        else:
            assert self.input_shape is not None

        assert cp.prod(self.input_shape[1:]) == cp.prod(self.output_shape[1:]), "can not change the element's number"
        Layer.connect(self, prev_layer)



    def __call__(self, layer: Layer)-> Layer:
        if layer is not None:
            self.input_shape = layer.output_shape
        else:
            assert self.input_shape is not None
        assert cp.prod(self.input_shape[1:]) == cp.prod(self.output_shape[1:]),"can not change the element's number"
        super(Reshape, self).__call__(layer)

        return self



    def forward(self, is_training: bool):
        N = self.input_tensor.shape[0]
        self.output_shape = (N,) + self.output_shape[1:]
        self.output_tensor = self.input_tensor.reshape(self.output_shape)
        if is_training:
            self.input_shape = (N, ) + self.input_shape[1:]
        super(Reshape,self).forward(is_training)



    def backward(self):
        for layer in self.inbounds:
            if layer.require_grads:
                layer.grads += cp.reshape(self.grads, self.input_shape)
            else:
                layer.grads = self.grads