# Ref: https://machinelearningmastery.com/how-to-use-an-encoder-decoder-lstm-to-echo-sequences-of-random-integers/
from numpy import array
from numpy import argmax
from pandas import DataFrame
from pandas import concat

# one hot encode sequence
def one_hot_encode(sequence, n_unique=50): #0-49 of lottery number, though 0 is not used
    encoding = list()
    #print (sequence.tail(1))
    #for row in sequence.iterrows(): #read each lottery drawresult (this is for dataframe)
    for row in sequence: #read each lottery draw result
        #print ('Row:', row) 
        encoding_row = list()
        for value_str in row:
            #print('Value: ', value_str)
            value = int(value_str) #the csv read its as str, not int
            #if isinstance (value, int): #ignore if not int              
            vector = [0 for _ in range(n_unique)]
            vector[value] = 1
            #print('Vector: ', vector)
            encoding_row.append(vector)
        #print('Encoding Row: ', encoding_row)
        encoding.append(encoding_row)
    return array(encoding)
 
# decode a one hot encoded string
def one_hot_decode(encoded_seq):
    return [argmax(vector) for vector in encoded_seq]

# convert encoded sequence to supervised learning
def to_supervised(sequence, n_in, n_out):
    # create lag copies of the sequence
    df = DataFrame(sequence)
    df = concat([df.shift(n_in-i-1) for i in range(n_in)], axis=1)
    # drop rows with missing values
    df.dropna(inplace=True)
    # specify columns for input and output pairs
    values = df.values
    width = sequence.shape[1]
    X = values.reshape(len(values), n_in, width)
    y = values[:, 0:(n_out*width)].reshape(len(values), n_out, width)

# 由2002年7月4日開始至今為止之攪珠結果 (http://mark6.groupsbuy.com/src/download.php)
# All 攪珠結果 http://mark6.groupsbuy.com/src/download.php
import csv
import requests
import pandas as pd 

download = requests.get('http://mark6.groupsbuy.com/data/markSixData.csv')
decoded_content = download.content.decode('utf-8')
marksix_csv = csv.reader(decoded_content.splitlines(), delimiter=',')
dataset = list()
for marksix_result in marksix_csv:
    #print(marksix_result) #each row of the csv file
    marksix_result = marksix_result[2:9] #remove 1st 2col => the year and draw series number and end with the 7 drawed numbers
    dataset.append(marksix_result)
dataset.pop(0) # remove the 1st row, ['Year', 'Draw number', '1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '']
#print(dataset[0]) #2-dim

# Below datafame wont work with the encoding
#marksix_results = pd.read_csv('http://mark6.groupsbuy.com/data/markSixData.csv')
#print(marksix_results.head())

# drop all colums and only remain 1st - 7th numbers
#dataset = marksix_results
#dataset.drop('Unnamed: 9', axis=1, inplace=True)
#dataset.drop('Year', axis=1, inplace=True)
#dataset.drop ('Draw number', axis=1, inplace=True)
#print(dataset.tail(1))

# one hot encode
encoded = one_hot_encode(dataset)

print("length of dataset(total lottery draws): ", len(encoded))
print('Last Lottery result:')
print('-> 1-hot decoded: ', one_hot_decode(encoded[len(encoded)-1]))
print('-> 1-hot encoded: ',encoded[len(encoded)-1])

# prepare data for the LSTM
#def get_data(n_in, n_out):
import numpy as np

def get_data(dataset, look_back):
    dataX, dataY = [], []
    #print(one_hot_decode(dataset[0]), len(dataset))
    for i in range (len(dataset)-1):
        # already one hot encode
        X_encoded_sequence = dataset[i]
        Y_encoded_sequence = dataset[i + look_back]
        #print('X_encoded_sequence', one_hot_decode(X_encoded_sequence))
        #print('Y_encoded_sequence', one_hot_decode(Y_encoded_sequence))
        X_df = DataFrame(X_encoded_sequence)
        Y_df = DataFrame(Y_encoded_sequence)
        #df = concat([df.shift(n_in-i-1) for i in range(n_in)], axis=1)
        # specify columns for input and output pairs
        X_values = X_df.values
        Y_values = Y_df.values
        #width = dataset.shape[1]
        #width = 7
        #print('width', width)
        #print('X_values', one_hot_decode(X_values))
        #print('Y_values', one_hot_decode(Y_values))
        #X = X_values.reshape(len(X_values), 1, width)
        #Y = Y_values.reshape(len(Y_values), 1, width)
        dataX.append(X_values)
        dataY.append(Y_values)
    return np.array(dataX), np.array(dataY)

# split into train and test sets -> 9.9:0.1
train_size = int(len(encoded) * 0.99)
test_size = len(encoded) - train_size
train, test = encoded[0:train_size,:], encoded[train_size:len(encoded),:]
print("length of dataset(total num of lottery draws): ", len(encoded), "\nlength of train dataset: ", len(train), "\nlength of test dataset: ", len(test))

# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = get_data(train, look_back)
testX, testY = get_data(test, look_back)

# The LSTM network expects the input data (X) to be provided with a specific array structure in the form of: [samples, time steps, features].
# Currently, our data is in the form: [samples, features]
print("Num of Training Sample(trainX): ", trainX.shape[0], "Num of feature (lottery result numbers): ", trainX.shape[1], trainX.shape[2])
print("Num of Testing Sample(testX): ", testX.shape[0], "Num of feature (lottery result numbers): ", testX.shape[1], testX.shape[2])

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
import tensorflow as tf

#GPU
config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

np.random.seed(7) 

# define LSTM
n_in = 7
n_out = 1
encoded_length = 50 #(0-49)
# https://www.mathsisfun.com/greatest-common-factor-tool.html 
# http://www.mathwarehouse.com/arithmetic/factorization-calculator.php
#batch_size = 82 #82x29=2378 all data
#batch_size = 104 #104x16=1664 training data (2, 4, 8, 13, 16, 26, 32, 52, 64, 104, 128, 208, 416, 832)
batch_size = 124 #put 1 for 1-step predict, not n-step
model = Sequential()
model.add(LSTM(600, batch_input_shape=(batch_size, n_in, encoded_length), return_sequences=True, stateful=True))
model.add(TimeDistributed(Dense(encoded_length, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# train LSTM
for epoch in range(5000):
    print('n_epoch: ', epoch)
    # generate new random sequence
    #X,y = get_data(encoded, 1)
    #X, y = get_data(train, look_back)
    # fit model for one epoch on this sequence
    #model.fit(X, y, epochs=5, batch_size=batch_size, verbose=2, shuffle=False)
    # this is for 1-step predict
    #model.fit(trainX, trainY, epochs=5, batch_size=batch_size, verbose=2, shuffle=False, validation_data=(testX, testY))
    # this is for n-step predict, to ocnvert to 1-step for testing need to copy the weight
    model.fit(trainX, trainY, epochs=15, batch_size=batch_size, verbose=2, shuffle=False)
    model.reset_states()
'''
# evaluate LSTM
test_batch_size = 1
predict = model.predict(testX, test_batch_size, verbose=1)
# decode all pairs
for i in range(len(testX)):
    print('test(X): ', one_hot_decode(test_X[i]))
    print('Expected(Y):', one_hot_decode(testY[i]), 'Predicted(Y)', one_hot_decode(predict[i]))

#print(one_hot_decode(testX[len(testX)-1]))
#print(one_hot_decode(predict[len(testX)-1]))
'''
# re-define the batch size
n_batch = 1
new_encoded_length = 50
# re-define model
new_model = Sequential()
new_model.add(LSTM(600, batch_input_shape=(n_batch, n_in, new_encoded_length), return_sequences=True, stateful=True))
new_model.add(TimeDistributed(Dense(encoded_length, activation='softmax')))

# copy weights
old_weights = model.get_weights()
new_model.set_weights(old_weights)
# compile model
new_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# online forecast
for i in range(len(testX)):
    test_X, test_Y = testX[i], testY[i]
    #print('X: ', one_hot_decode(test_X), 'Y: ', one_hot_decode(test_Y))
    test_X = test_X.reshape(1, 7, 50)
    #yhat = new_model.predict(test_X, batch_size=n_batch)
    print('testX: ', one_hot_decode(test_X[0]))
    yhat = new_model.predict(test_X)
    print('>Expected= ',one_hot_decode(test_Y), ' Predict= ', one_hot_decode(yhat[0]))
    
# Next draw predict    
this_sequence = one_hot_encode(np.array([['13', '14', '15', '37', '38', '44', '40'], ['13', '14', '15', '37', '38', '44', '40']]))          
this_X_1 = DataFrame(this_sequence[0])
X_value_1 = this_X_1.values
print('X_value: ', one_hot_decode(X_value_1))
X_value_1 = X_value_1.reshape(1, 7, 50)
predict_Y_1 = new_model.predict(X_value_1)
print('Next draw Predict= ', one_hot_decode(predict_Y_1[0]))

this_X_2 = DataFrame(this_sequence[1])
X_value_2 = this_X_2.values
print('X_value: ', one_hot_decode(X_value_2))
X_value_2 = X_value_2.reshape(1, 7, 50)
predict_Y_2 = new_model.predict(X_value_2)
print('Next draw Predict= ', one_hot_decode(predict_Y_2[0]))
