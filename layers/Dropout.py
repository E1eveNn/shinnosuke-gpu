import cupy as cp

from .Base import Layer


class Dropout(Layer):
    def __init__(self,keep_prob):
        #prob :probability of keeping a unit active.
        self.keep_prob=keep_prob
        super(Dropout,self).__init__()



    def connect(self,prev_layer):
        assert 0.<self.keep_prob<1.
        self.output_shape=prev_layer.output_shape
        Layer.connect(self,prev_layer)


    def __call__(self, prev_layer):
        super(Dropout,self).__call__(prev_layer)
        self.output_shape=self.input_shape
        return self


    def forward(self,is_training=True):
        if is_training:
            random_tensor=cp.random.binomial(n=1,p=self.keep_prob,size=self.input_tensor.shape)
            self.output_tensor=self.input_tensor*random_tensor
            self.output_tensor/=self.keep_prob


            if self.require_grads:
                self.mask = random_tensor
                self.grads=cp.zeros_like(self.output_tensor)
        else:
            self.output_tensor=self.input_tensor
        super().forward(is_training)


    def backward(self):
        for layer in self.inbounds:
            if layer.require_grads:
                layer.grads+=(self.grads*self.mask/self.keep_prob)
            else:
                layer.grads=self.grads


