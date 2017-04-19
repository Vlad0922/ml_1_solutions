import sys
import os
import gc

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.wrappers.scikit_learn import KerasClassifier

import matplotlib.pyplot as plt
import seaborn


def train_val_test_split(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size = 0.8, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, train_size=0.5, stratify=y_val)
    
    return X_train, y_train, X_val, y_val, X_test, y_test  


def create_model_simple(activation):
    model = Sequential()

    model.add(Flatten(input_shape=(3, 28, 28)))
    model.add(Dense(4096, activation=activation))
    model.add(Dense(2048, activation=activation))
    model.add(Dense(1024, activation=activation))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

def create_model_complex(activation):
    model = Sequential()

    model.add(Conv2D(3, (3, 3), input_shape=(28, 28, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(3, (3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(4096, activation=activation))
    model.add(Dense(2048, activation=activation))
    model.add(Dense(1024, activation=activation))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

def create_model_dropout(drop):
    model = Sequential()

    model.add(Conv2D(3, (3, 3), input_shape=(28, 28, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(3, (3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(4096, activation='sigmoid'))
    model.add(Dropout(drop))
    model.add(Dense(2048, activation='sigmoid'))
    model.add(Dropout(drop))
    model.add(Dense(1024, activation='sigmoid'))
    model.add(Dropout(drop))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model


data = list()
labels = list()
for char in os.listdir('data/notMNIST_small/'):
    for fname in os.listdir('data/notMNIST_small/' + char):
        img = load_img('data/notMNIST_small/' + char + '/' + fname)
        data.append(img_to_array(img))
        labels.append(char)
data = np.array(data)
labels = np.array(labels)

# X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(data, labels)

def run_model_simple(activation):
    model = KerasClassifier(build_fn=lambda: create_model_simple(activation), epochs=10, batch_size=500, validation_split=0.25, verbose=1)
    history = model.fit(data, labels)
    
    plt.figure(figsize=(15,10))
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('result/simple_accuracy_'+activation+'.png')

    plt.figure(figsize=(15,10))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('result/simple_loss_'+activation+'.png')

def run_model_complex(activation):
    model = KerasClassifier(build_fn=lambda: create_model_complex(activation), epochs=10, batch_size=500, validation_split=0.25, verbose=1)
    history = model.fit(data, labels)
    
    plt.figure(figsize=(15,10))
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('result/complex_accuracy_'+activation+'.png')

    plt.figure(figsize=(15,10))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('result/complex_loss_'+activation+'.png')


#well...
def find_dropout(d_range):
    with open('dump.txt', 'a') as f:
        res = list()
        for d in d_range:
            X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size = 0.8, random_state=0)
            model = KerasClassifier(build_fn=lambda: create_model_dropout(d), epochs=10, batch_size=500, validation_split=0.25, verbose=1)
            model.fit(X_train, y_train)

            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            res.append(acc)
            print 'Dropout: {}, accuracy: {}'.format(d, acc)
            del model
            gc.collect(0)
            gc.collect(1)
            gc.collect(2)
            f.write('{}:{}\n'.format(d, acc))

    plt.figure(figsize=(15,10))
    plt.plot(d_range, res)
    plt.ylabel('Accuracy')
    plt.xlabel('droptout rate')
    plt.savefig('result/dropout_accuracy.png')

    best_idx = np.argmax(res)
    print 'best accuracy: {}, with dropout: {}'.format(res[best_idx], d_range[best_idx])
    


# run_model_simple('tanh')
# run_model_simple('relu')
# run_model_simple('sigmoid')
# run_model_complex('tanh')
# run_model_complex('relu')
# run_model_complex('sigmoid')
find_dropout(np.arange(0.4, 0.55, 0.05))