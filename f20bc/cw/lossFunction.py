import numpy as np

class lossFunction:
    def evaluate(self,y_pred,y_train):
        self.y_pred=y_pred
        self.y_train=y_train

class MeanAbsoluteError(lossFunction):
    def evaluate(self, y_pred, y_train):
        f= np.mean(np.abs(y_pred - y_train))
        return f
