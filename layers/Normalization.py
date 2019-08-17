import cupy as cp
from functools import reduce
from .Base import Layer, Variable
from ..utils.Initializers import get_initializer



class BatchNormalization(Layer):
    def __init__(self,epsilon=1e-6,momentum=0.9,axis=1,gamma_initializer='ones',beta_initializer='zeros',moving_mean_initializer='zeros', moving_variance_initializer='ones'):
        # axis=1 when input Fully Connected Layers(data shape:(M,N),where M donotes Batch-size,and N represents feature nums)  ---also axis=-1 is the same
        # axis=1 when input Convolution Layers(data shape:(M,C,H,W),represents Batch-size,Channels,Height,Width,respectively)

        self.epsilon=epsilon
        self.axis=axis
        self.momentum=momentum
        self.gamma_initializer=get_initializer(gamma_initializer)
        self.beta_initializer=get_initializer(beta_initializer)
        self.moving_mean_initializer=get_initializer(moving_mean_initializer)
        self.moving_variance_initializer=get_initializer(moving_variance_initializer)
        super(BatchNormalization,self).__init__()



    def connect(self,prev_layer):
        n_in=prev_layer.output_shape[self.axis]
        gamma = Variable(self.gamma_initializer(n_in))
        beta = Variable(self.beta_initializer(n_in))
        gamma.grads = cp.zeros_like(gamma.output_tensor) if gamma.require_grads else None
        beta.grads = cp.zeros_like(beta.output_tensor) if beta.require_grads else None
        self.variables.append(gamma)
        self.variables.append(beta)
        self.moving_mean=self.moving_mean_initializer(n_in)
        self.moving_variance=self.moving_variance_initializer(n_in)
        self.output_shape=prev_layer.output_shape

        Layer.connect(self, prev_layer)



    def __call__(self,prev_layer):
        super(BatchNormalization,self).__call__(prev_layer)
        n_in = self.input_shape[self.axis]
        gamma = Variable(self.gamma_initializer(n_in))
        beta = Variable(self.beta_initializer(n_in))
        gamma.grads = cp.zeros_like(gamma.output_tensor) if gamma.require_grads else None
        beta.grads = cp.zeros_like(beta.output_tensor) if beta.require_grads else None
        self.variables.append(gamma)
        self.variables.append(beta)
        self.moving_mean = self.moving_mean_initializer(n_in)
        self.moving_variance = self.moving_variance_initializer(n_in)
        self.output_shape=self.input_shape
        return self


    # def forward(self,is_training=True):
    #     inputs=self.input_tensor
    #     gamma,beta=self.variables
    #     outputs = cp.zeros_like(inputs)
    #     if is_training:
    #         self.cache = []
    #         for k in range(inputs.shape[self.axis]):
    #         #calc mean
    #             mean=cp.mean(inputs[:,k])
    #             #calc var
    #             var=cp.var(inputs[:,k])
    #             #x minus u
    #             xmu=inputs[:,k]-mean
    #             sqrtvar=cp.sqrt(var+self.epsilon)
    #             normalized_x=xmu/sqrtvar
    #             outputs[:,k]=gamma.output_tensor[k]*normalized_x+beta.output_tensor[k]
    #             self.cache.append((xmu,sqrtvar,normalized_x))
    #             self.moving_mean[k]=self.momentum*self.moving_mean[k]+(1-self.momentum)*mean
    #             self.moving_variance[k] = self.momentum * self.moving_variance[k] + (1 - self.momentum) * var
    #
    #     else:
    #         for k in range(inputs.shape[self.axis]):
    #             std=cp.sqrt(self.moving_variance[k]+self.epsilon)
    #             outputs[:,k]=(gamma.output_tensor[k]/std)*inputs[:,k]+(beta.output_tensor[k]-gamma.output_tensor[k]*self.moving_mean[k]/std)
    #
    #
    #     self.output_tensor=outputs
    #     if self.require_grads:
    #         self.grads = cp.zeros_like(self.output_tensor)
    #     super().forward(is_training)






    def forward(self,is_training=True):
        inputs=self.input_tensor
        self.input_shape=inputs.shape
        if self.input_tensor.ndim==4:
            N,C,H,W=self.input_shape
            inputs=inputs.transpose(0,3,2,1).reshape(N*H*W,C)


        gamma,beta=self.variables
        if is_training:

            #calc mean
            mean=cp.mean(inputs,axis=0)
            #calc var
            var=cp.var(inputs,axis=0)
            #x minus u
            xmu=inputs-mean
            sqrtvar=cp.sqrt(var+self.epsilon)
            normalized_x=xmu/sqrtvar
            outputs=gamma.output_tensor*normalized_x+beta.output_tensor
            self.cache=(xmu,sqrtvar,normalized_x)
            self.moving_mean=self.momentum*self.moving_mean+(1-self.momentum)*mean
            self.moving_variance = self.momentum * self.moving_variance + (1 - self.momentum) * var

        else:
            scale = gamma.output_tensor / (cp.sqrt(self.moving_variance + self.epsilon))
            outputs = inputs * scale + (beta.output_tensor - self.moving_mean * scale)
        if self.input_tensor.ndim==4:
            N, C, H, W = self.input_shape
            outputs=outputs.reshape(N, W, H, C).transpose(0, 3, 2, 1)
        self.output_tensor=outputs

        if self.require_grads:
            self.grads = cp.zeros_like(self.output_tensor)
        super().forward(is_training)


    # def backward(self):
    #     grads=self.grads
    #     gamma,beta=self.variables
    #     outputs=cp.zeros_like(grads)
    #     for k in range(grads.shape[self.axis]):
    #         xmu,sqrtvar,normalzied_x=self.cache[k]
    #         if beta.require_grads:
    #             beta.grads[k]+=cp.sum(grads[:,k])
    #         if gamma.require_grads:
    #             gamma.grads[k]+=cp.sum(grads[:,k]*normalzied_x)
    #
    #         dnormalized_x=grads[:,k]*gamma.output_tensor[k]
    #         # equals to var^-3/2,where sqrtvar=var^1/2
    #         dvar=cp.sum(cp.power(-1./sqrtvar,3)*xmu*dnormalized_x*0.5)
    #
    #         dmean=cp.sum(-dnormalized_x/sqrtvar)-dvar*2*cp.mean(xmu)
    #         m=cp.prod(cp.asarray(xmu.shape)).tolist()
    #         outputs[:,k]=dnormalized_x/sqrtvar+dvar*2*xmu/m+dmean/m
    #     for layer in self.inbounds:
    #         if layer.require_grads:
    #             layer.grads+=outputs
    #         else:
    #             layer.grads=grads



    def backward(self):
        grads=self.grads
        if len(self.input_shape)==4:
            N, C, H, W = self.input_shape
            grads=grads.transpose(0, 3, 2, 1).reshape((N * H * W, C))

        gamma,beta=self.variables
        xmu, sqrtvar, normalized_x = self.cache
        if beta.require_grads:
            beta.grads += cp.sum(grads,axis=0)
        if gamma.require_grads:
            gamma.grads += cp.sum(grads * normalized_x,axis=0)

        N=normalized_x.shape[0]
        dnormalized_x = grads * gamma.output_tensor
        dvar = cp.sum(cp.power( - 1./sqrtvar, 3) * xmu * dnormalized_x * 0.5, axis=0)
        dmean=cp.sum( - dnormalized_x / sqrtvar, axis=0) - 2 * dvar * cp.mean(xmu, axis=0)
        outputs=dnormalized_x / sqrtvar + dvar * 2 * xmu / N + dmean / N
        if len(self.input_shape)==4:
            N, C, H, W = self.input_shape
            outputs = outputs.reshape(N, W, H, C).transpose(0, 3, 2, 1)
        for layer in self.inbounds:
            if layer.require_grads:
                layer.grads+=outputs
            else:
                layer.grads=grads




class LayerNormalization(Layer):
    def __init__(self,epsilon=1e-10,gamma_initializer='ones',beta_initializer='zeros'):

        self.epsilon=epsilon
        self.gamma_initializer=get_initializer(gamma_initializer)
        self.beta_initializer=get_initializer(beta_initializer)
        super(LayerNormalization,self).__init__()



    def connect(self,prev_layer):
        normal_shape=prev_layer.output_shape[1:]
        gamma = Variable(self.gamma_initializer(normal_shape))
        beta = Variable(self.beta_initializer(normal_shape))
        gamma.grads = cp.zeros_like(gamma.output_tensor) if gamma.require_grads else None
        beta.grads = cp.zeros_like(beta.output_tensor) if beta.require_grads else None
        self.variables.append(gamma)
        self.variables.append(beta)

        self.output_shape=prev_layer.output_shape

        Layer.connect(self, prev_layer)



    def __call__(self,prev_layer):
        super(LayerNormalization,self).__call__(prev_layer)
        normal_shape=prev_layer.output_shape[1:]
        gamma = Variable(self.gamma_initializer(normal_shape))
        beta = Variable(self.beta_initializer(normal_shape))
        gamma.grads = cp.zeros_like(gamma.output_tensor) if gamma.require_grads else None
        beta.grads = cp.zeros_like(beta.output_tensor) if beta.require_grads else None
        self.variables.append(gamma)
        self.variables.append(beta)

        self.output_shape=self.input_shape
        return self



    def forward(self,is_training=True):
        inputs=self.input_tensor
        # self.input_shape =inputs.shape
        self.shape_field=tuple([i for i in range(1,inputs.ndim)])
        gamma,beta=self.variables

        #calc mean

        mean=cp.mean(inputs,axis=self.shape_field,keepdims=True)
        #calc var
        var=cp.mean(inputs,axis=self.shape_field,keepdims=True)

        #x minus u
        xmu=inputs-mean
        sqrtvar=cp.sqrt(var+self.epsilon)
        normalized_x=xmu/sqrtvar
        outputs=gamma.output_tensor*normalized_x+beta.output_tensor
        self.cache=(xmu,sqrtvar,normalized_x)

        self.output_tensor=outputs

        if self.require_grads:
            self.grads = cp.zeros_like(self.output_tensor)
        super().forward(is_training)






    def backward(self):
        grads=self.grads
        xmu, sqrtvar, normalized_x = self.cache
        std_inv=1./sqrtvar

        N=reduce(lambda x,y:x*y,normalized_x.shape[1:])
        gamma,beta=self.variables

        if beta.require_grads:
            beta.grads += cp.sum(grads,axis=0)
        if gamma.require_grads:
            gamma.grads += cp.sum(grads * normalized_x,axis=0)

        dnormalized_x = grads * gamma
        dvar = (-0.5) * cp.sum(dnormalized_x * xmu, axis=self.shape_field, keepdims=True) * (std_inv ** 3) #(m,1)=(m,c,h,w)*(m,c,h,w)*(m,1)

        dmean = (-1.0) * cp.sum(dnormalized_x * std_inv, axis=self.shape_field, keepdims=True) - 2.0 * dvar * cp.mean(xmu, axis=self.shape_field,keepdims=True)



        outputs = dnormalized_x * std_inv + (2. / N) * dvar * xmu + (1. / N) * dmean

        for layer in self.inbounds:
            if layer.require_grads:
                layer.grads+=outputs
            else:
                layer.grads=grads





class GroupNormalization(Layer):
    def __init__(self,epsilon=1e-5,G=16,gamma_initializer='ones',beta_initializer='zeros'):

        self.epsilon=epsilon
        self.G=G
        self.gamma_initializer=get_initializer(gamma_initializer)
        self.beta_initializer=get_initializer(beta_initializer)
        super(GroupNormalization,self).__init__()



    def connect(self,prev_layer):
        C=prev_layer.output_shape[1]
        assert C % self.G == 0
        gamma = Variable(self.gamma_initializer((1,C,1,1)))
        beta = Variable(self.beta_initializer((1,C,1,1)))
        gamma.grads = cp.zeros_like(gamma.output_tensor) if gamma.require_grads else None
        beta.grads = cp.zeros_like(beta.output_tensor) if beta.require_grads else None
        self.variables.append(gamma)
        self.variables.append(beta)

        self.output_shape=prev_layer.output_shape

        Layer.connect(self, prev_layer)



    def __call__(self,prev_layer):
        super(GroupNormalization,self).__call__(prev_layer)
        C = prev_layer.output_shape[1]
        assert C%self.G==0
        gamma = Variable(self.gamma_initializer((1, C, 1, 1)))
        beta = Variable(self.beta_initializer((1, C, 1, 1)))
        gamma.grads = cp.zeros_like(gamma.output_tensor) if gamma.require_grads else None
        beta.grads = cp.zeros_like(beta.output_tensor) if beta.require_grads else None
        self.variables.append(gamma)
        self.variables.append(beta)

        self.output_shape=self.input_shape
        return self



    def forward(self,is_training=True):
        inputs=self.input_tensor
        # self.input_shape =inputs.shape
        gamma,beta=self.variables
        N, C, H, W = inputs.shape
        self.shape_field=tuple([i for i in range(2,inputs.ndim)])


        x_group = cp.reshape(inputs, (N, self.G, C // self.G, H, W))
        mean = cp.mean(x_group, axis=self.shape_field, keepdims=True)
        var = cp.var(x_group, axis=self.shape_field, keepdims=True)
        xgmu=x_group - mean
        sqrtvar=cp.sqrt(var + self.epsilon)
        x_group_norm = xgmu / sqrtvar

        x_norm = cp.reshape(x_group_norm, (N, C, H, W))

        outputs = gamma.output_tensor * x_norm + beta.output_tensor

        self.cache = ( xgmu, sqrtvar, x_norm)

        self.output_tensor=outputs

        if self.require_grads:
            self.grads = cp.zeros_like(self.output_tensor)
        super().forward(is_training)




    def backward(self):
        grads=self.grads
        xgmu, sqrtvar, x_norm = self.cache

        std_inv=1./sqrtvar

        gamma,beta=self.variables

        N, C, H, W = grads.shape
        if beta.require_grads:
            beta.grads += cp.sum(grads, axis=(0, 2, 3), keepdims=True)
        if gamma.require_grads:
            gamma.grads += cp.sum(grads * x_norm, axis=(0, 2, 3), keepdims=True)

        # dx_group_norm
        dx_norm = grads * gamma  #(N,C,H,W)
        dx_group_norm = cp.reshape(dx_norm, (N, self.G, C // self.G, H, W))
        # dvar
        dvar = -0.5 * (std_inv ** 3) * cp.sum(dx_group_norm * xgmu, axis=(2, 3, 4), keepdims=True)
        # dmean
        N_GROUP = C // self.G * H * W
        dmean1 = cp.sum(dx_group_norm * -std_inv, axis=(2, 3, 4), keepdims=True)
        dmean2_var = dvar * -2.0 / N_GROUP * cp.sum(xgmu, axis=(2, 3, 4), keepdims=True)
        dmean = dmean1 + dmean2_var
        # dx_group
        dx_group1 = dx_group_norm * std_inv
        dx_group_var = dvar * 2.0 / N_GROUP * xgmu
        dx_group_mean = dmean * 1.0 / N_GROUP
        dx_group = dx_group1 + dx_group_var + dx_group_mean
        # dx
        outputs = cp.reshape(dx_group, (N, C, H, W))



        for layer in self.inbounds:
            if layer.require_grads:
                layer.grads+=outputs
            else:
                layer.grads=grads



