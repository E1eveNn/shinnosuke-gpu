import cupy as cp
import gc

from ..layers.Base import Operation


class Relu(Operation):
    def _forward(self,inputs,is_training=True):
        if is_training:
            self.inputs=inputs
        return cp.maximum(0,inputs)



    def forward(self,is_training=True):
        return self._forward(self.input_tensor,is_training)



    def _backward(self,grad):
        grad[self.inputs < 0] = 0
        del self.inputs
        gc.collect()
        return grad


    def backward(self):
        return self._backward(self.grads)



class Linear(Operation):
    def _forward(self,inputs,is_training=True):
        return inputs


    def forward(self,is_training):
        return self._forward(self.input_tensor,is_training)



    def _backward(self,grad):
        return grad


    def backward(self):
        return self._backward(self.grads)




class Sigmoid(Operation):
    def _forward(self,inputs,is_training=True):
        output=1/(1+cp.exp(-inputs))
        if is_training:
            self.output=output
        return output



    def forward(self,is_training):
        return self._forward(self.input_tensor,is_training)



    def _backward(self,grad):
        return grad*self.output*(1-self.output)



    def backward(self):
        return self._backward(self.grads)



class Tanh(Operation):
    def _forward(self,inputs,is_training=True):
        output=cp.tanh(inputs)
        if is_training:
            self.output=output
        return output



    def forward(self,is_training):
        return self._forward(self.input_tensor,is_training)


    def _backward(self,grad):
        return grad*(1-cp.square(self.output))


    def backward(self):
        return self._backward(self.grads)






class Softmax(Operation):

    #more stable softmax
    def _forward(self,inputs,is_training=True):
        shiftx = inputs - cp.max(inputs)
        outputs= cp.divide(cp.exp(shiftx),cp.sum(cp.exp(shiftx),axis=-1,keepdims=True))
        del inputs
        if is_training:
            self.outputs=outputs
        return outputs

    def forward(self, is_training):
        return self._forward(self.input_tensor, is_training)



    def _backward(self,grad):
        del self.outputs
        gc.collect()
        return grad



    def backward(self):
        return self._backward(self.grads)













def get_activator(activator):
    if activator.__class__.__name__=='str':
        activator=activator.lower()
        if activator=='relu':
            return Relu()
        elif activator=='softmax':
            return Softmax()
        elif activator=='linear':
            return Linear()
        elif activator=='sigmoid':
            return Sigmoid()
        elif activator=='tanh':
            return Tanh()
