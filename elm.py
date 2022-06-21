import numpy as np
import pandas as pd
import scipy
import time
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from codecarbon import EmissionsTracker
train = pd.read_csv('mnist_train.csv')
test = pd.read_csv('mnist_test.csv')

onehotencoder = OneHotEncoder(categories='auto')
scaler = MinMaxScaler()
X_train = scaler.fit_transform(train.values[:,1:])
y_train = onehotencoder.fit_transform(train.values[:,:1]).toarray()
X_test = scaler.fit_transform(test.values[:,1:])
y_test = onehotencoder.fit_transform(test.values[:,:1]).toarray()
input_size = X_train.shape[1]
hidden_size = 5000
input_weights = np.random.normal(size=[input_size,hidden_size])
biases = np.random.normal(size=[hidden_size])

def sigmoid(x):
    sig = np.where(x < 0, np.exp(x) / (1 + np.exp(x)), 1 / (1 + np.exp(-x)))
    return sig

def hidden_nodes(X):
   G = np.dot(X, input_weights)
   G = G + biases
   H = sigmoid(G)
   return H
print("ELM")
start = time.time()
#we need to divede to elms in order to get metaelm to work
tracker = EmissionsTracker()
tracker.start()
output_weights = np.dot(scipy.linalg.pinv(hidden_nodes(X_train)), y_train) # traning of extreme learning machine
emissions: float = tracker.stop()
end = time.time()

#startmeta = time.time() # although this seems that we should use it,actualy neednt
def predict(X):
   out = hidden_nodes(X)
   out = np.dot(out, output_weights)
   return out

prediction = predict(X_test)
correct = 0
false = 0
total = X_test.shape[0]
for i in range(total):
    predicted = np.argmax(prediction[i])
    actual = np.argmax(y_test[i])
    #correct += 1 if predicted == actual else 0
    if predicted == actual:
        correct += 1
    else: false += 1

loss = false/total
accuracy = correct/total
print('Accuracy for ', hidden_size, ' hidden nodes: ', accuracy)
print('Loss for ', hidden_size, ' hidden nodes: ', loss)
print('Time for ELM',end - start, ' seconds')
print("%.4f" % (emissions*1000) + " gram CO\u2082")