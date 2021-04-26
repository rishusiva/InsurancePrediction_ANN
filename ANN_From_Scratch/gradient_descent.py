from NN_model import *
import math
import numpy as np

coef,intercept = model.get_weights()

def sigmoid(X):
    return 1/ (1+math.exp(-X))


def prediction_function(age,affordibility):
    weighted_sum = coef[0]*age + coef[1]*affordibility +  intercept
    return sigmoid(weighted_sum)


#print(prediction_function(.28,1))

def loss_function(y_true,y_predicted):
    epsilon=1e-15
    y_predicted_new = [max(i,epsilon) for i in y_predicted]
    y_predicted_new = [min(i,1-epsilon) for i in y_predicted_new]
    y_predicted_new = np.array(y_predicted_new)
    return -np.mean(y_true*np.log(y_predicted_new)+(1-y_true)*np.log(1-y_predicted_new))


def sigmoid_numpy(X):
   return 1/(1+np.exp(-X))


def gradient_descent(age,affordibility,y_true,epochs,loss_threshold):
    w1 = w2 = 1
    b = 0
    learning_rate = 0.5
    m = len(age)

    for i in range(epochs):
        weighted_sum = w1*age + w2*affordibility + b
        y_predicted = sigmoid_numpy(weighted_sum)

        loss = loss_function(y_true,y_predicted)

        dw1 = (1/m)*np.dot(np.transpose(age),(y_predicted-y_true))
        dw2 = (1/m)*np.dot(np.transpose(affordibility),(y_predicted-y_true))
        db = np.mean(y_predicted-y_true)


        w1 = w1 - learning_rate*dw1
        w2 = w2 - learning_rate*dw2
        b = b - learning_rate*db

        print(f'Epoch:{i},w1:{w1},w2:{w2},bias:{b},loss:{loss}')

        if loss<=loss_threshold:
            break
    return w1, w2 , b

print(gradient_descent(X_train_scaled['age'],X_train_scaled['affordibility'],y_train,1000,0.4631))