import cupy as cp

import copy


class Objective():
    pass




class MeanSquaredError():



    def calc_loss(self,y_hat,y):
        return 0.5*cp.mean(cp.sum(cp.power(y_hat-y,2),axis=1))



    def backward(self,y_hat,y):
        return y_hat-y



class MeanAbsoluteError(Objective):
    def __init__(self):
        super(MeanAbsoluteError, self).__init__('linear')



    def calc_loss(self, y_hat, y):
        return cp.mean(cp.sum(cp.absolute(y_hat - y), axis=1))


    def backward(self, y_hat, y):
        pos=cp.where((y_hat-y)<0)
        mask=cp.ones_like(y_hat)
        mask[pos]=-1
        return mask




class BinaryCrossEntropy(Objective):
    def __init__(self):
        super(BinaryCrossEntropy, self).__init__('sigmoid')


    def calc_loss(self,y_hat,y):
        loss=-cp.multiply(y,y_hat)-cp.multiply(1-y,cp.log(1-y_hat))
        return cp.mean(cp.sum(loss,axis=1))


    def backward(self,y_hat,y):
        return (y_hat-y)/ y_hat.shape[0]





class SparseCategoricalCrossEntropy(Objective):
    def calc_acc(self,y_hat,y):
        acc = (cp.argmax(y_hat, axis=-1) == cp.argmax(y, axis=-1))
        acc = cp.mean(acc).tolist()
        return acc



    def calc_loss(self,y_hat,y):
        avg=cp.prod(cp.asarray(y_hat.shape[:-1]))
        loss=-cp.sum(cp.multiply(y,cp.log(y_hat)))/avg
        return loss.tolist()





    def backward(self,y_hat,y_true):
        avg = cp.prod(cp.asarray(y_hat.shape[:-1]))
        return (y_hat-y_true)/avg







class CategoricalCrossEntropy(Objective):
    def calc_acc(self,y_hat,y):
        acc = (cp.argmax(y_hat, axis=-1,keepdims=True) == y)
        acc = cp.mean(acc).tolist()
        return acc

    def calc_loss(self,y_hat,y):
        to_sum_shape=cp.asarray(y_hat.shape[:-1])
        avg=cp.prod(to_sum_shape)
        idx=[]
        for s in to_sum_shape:
            idx.append(cp.arange(s).tolist())
        idx.append(y.flatten().tolist())

        loss=-cp.sum(cp.log(y_hat[idx]))/avg
        return loss.tolist()





    def backward(self,y_hat,y_true):
        to_sum_shape = cp.asarray(y_hat.shape[:-1])
        avg = cp.prod(to_sum_shape)
        idx = []
        for s in to_sum_shape:
            idx.append(cp.arange(s).tolist())
        idx.append(y_true.flatten().tolist())

        y_hat[idx]-=1
        return y_hat/avg






def get_objective(objective):
    if objective.__class__.__name__=='str':
        objective=objective.lower()
        if objective in['categoricalcrossentropy','categorical_crossentropy','categorical_cross_entropy']:
            return CategoricalCrossEntropy()
        elif objective in['sparsecategoricalcrossentropy','sparse_categorical_crossentropy','sparse_categorical_cross_entropy']:
            return SparseCategoricalCrossEntropy()
        elif objective in ['binarycrossentropy','binary_cross_entropy']:
            return BinaryCrossEntropy()
        elif objective in ['meansquarederror','mean_squared_error','mse']:
            return MeanSquaredError()
        elif objective in ['meanabsoluteerror','mean_absolute_error','mae']:
            return MeanAbsoluteError()
        elif isinstance(objective,Objective):
            return copy.deepcopy(objective)
        else:
            raise ValueError('unknown objective type!')
