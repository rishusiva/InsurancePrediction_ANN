from cleaning_dataset import *
import numpy as np

def sigmoid_numpy(X):
        return 1/(1+np.exp(-X))


def loss_function(y_true,y_predicted):
        epsilon=1e-15
        y_predicted_new = [max(i,epsilon) for i in y_predicted]
        y_predicted_new = [min(i,1-epsilon) for i in y_predicted_new]
        y_predicted_new = np.array(y_predicted_new)
        return -np.mean(y_true*np.log(y_predicted_new)+(1-y_true)*np.log(1-y_predicted_new))


class NN_from_scratch:
    def __init__(self):
        self.w1 = 1 
        self.w2 = 1
        self.b = 0
        
    def fit(self, X, y, epochs, loss_thresold):
        self.w1, self.w2, self.b = self.gradient_descent(X['age'],X['affordibility'],y, epochs, loss_thresold)
        print(f"Final weights and bias: w1: {self.w1}, w2: {self.w2}, bias: {self.b}")
        
    def predict(self, X_test):
        weighted_sum = self.w1*X_test['age'] + self.w2*X_test['affordibility'] + self.b
        return sigmoid_numpy(weighted_sum)


    def gradient_descent(self, age,affordability, y_true, epochs, loss_thresold):
        w1 = w2 = 1
        b = 0
        learning_rate = 0.5
        n = len(age)
        for i in range(epochs):
            weighted_sum = w1 * age + w2 * affordability + b
            y_predicted = sigmoid_numpy(weighted_sum)
            loss = loss_function(y_true, y_predicted)
            
            dw1 = (1/n)*np.dot(np.transpose(age),(y_predicted-y_true)) 
            dw2 = (1/n)*np.dot(np.transpose(affordability),(y_predicted-y_true)) 

            db = np.mean(y_predicted-y_true)
            w1 = w1 - learning_rate * dw1
            w2 = w2 - learning_rate * dw2
            b = b - learning_rate * db
            
            if i%50==0:
                print (f'Epoch:{i}, w1:{w1}, w2:{w2}, bias:{b}, loss:{loss}')
            
            if loss<=loss_thresold:
                print (f'Epoch:{i}, w1:{w1}, w2:{w2}, bias:{b}, loss:{loss}')
                break

        return w1, w2, b
    
Model = NN_from_scratch()
Model.fit(X_train_scaled, y_train, epochs=8000, loss_thresold=0.4631)

#print(Model.predict(X_test_scaled))
    