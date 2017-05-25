from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, TimeDistributed
from keras.layers import LSTM, SimpleRNN, GRU
from config import config

def build_model(input_shape):
    '''
        Sample LSTM Model
    '''

    model = Sequential()
    model.add(LSTM(config['length'], input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.25))
    model.add(LSTM(config['length'], return_sequences=True))
    model.add(Dropout(0.25))
    model.add(LSTM(config['length']))
    model.add(Dropout(0.25))
    model.add(Dense(6))
    model.add(Activation('sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
