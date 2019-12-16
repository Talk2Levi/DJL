
import sys
import numpy as np
import pandas as pd
import pydub
import pickle
import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras.layers import Dense, LSTM, LeakyReLU, Activation
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import optimizers
from scipy.io.wavfile import read, write

np.set_printoptions(threshold=sys.maxsize)

# converting mp3 to wave
def mp3_to_wav(path_to_mp3):
    sound = pydub.AudioSegment.from_mp3(path_to_mp3)
    wav_name = path_to_mp3+".wav"
    sound.export(wav_name, format="wav")
    return read(wav_name)

# function to create training data by shifting the music data 
def create_train_dataset(df, look_back, train=True):
    dataX1, dataX2 , dataY1 , dataY2 = [],[],[],[]
    for i in range(len(df)-look_back-1):
        dataX1.append(df.iloc[i : i + look_back, 0].values)
        dataX2.append(df.iloc[i : i + look_back, 1].values)
        if train:
            dataY1.append(df.iloc[i + look_back, 0])
            dataY2.append(df.iloc[i + look_back, 1])
    if train:
        return np.array(dataX1, dtype=np.float32), np.array(dataX2, dtype=np.float32), np.array(dataY1, dtype=np.float32), np.array(dataY2, dtype=np.float32)
    else:
        return np.array(dataX1, dtype=np.float32), np.array(dataX2, dtype=np.float32)

def train_portion(sample_num):
    return int(len(sample_num) * 0.2)

def get_train_data(music, file_path, portion = 0, load_existing_data=False):
    X1, X2, y1, y2 = [], [], [], []
    if load_existing_data:
        with open(file_path, 'rb') as f:
            X1, X2, y1, y2 = pickle.load(f)
    else:    
        X1, X2, y1, y2  = create_train_dataset(
            pd.DataFrame(music.iloc[:portion, :]),
            look_back=3,
            train=True)
        X1 = X1.reshape((-1, 1, 3))
        X2 = X2.reshape((-1, 1, 3))

        with open(file_path, 'wb') as f:
            pickle.dump([X1, X2, y1, y2], f)

    X1, X2, y1, y2 = tf.convert_to_tensor(X1,dtype=tf.float32), tf.convert_to_tensor(X2,dtype=tf.float32), tf.convert_to_tensor(y1,dtype=tf.float32), tf.convert_to_tensor(y2,dtype=tf.float32)
    return X1, X2, y1, y2

def get_test_data(music, file_path, portion = 0,load_existing_data = False):
    test1, test2 = [], []
    if load_existing_data:
        with open(file_path, 'rb') as f:
            test1, test2 = pickle.load(f)
    else:
        test1, test2 = create_train_dataset(
            pd.DataFrame(music.iloc[portion+1:, :]),
            look_back=3, 
            train=False)
        test1 = test1.reshape((-1, 1, 3))
        test2 = test2.reshape((-1, 1, 3))

        with open(file_path, 'wb') as f:
            pickle.dump([test1, test2], f)

    test1, test2 = tf.convert_to_tensor(test1,dtype=tf.float32), tf.convert_to_tensor(test2,dtype=tf.float32)
    return test1, test2
    
# music1_train, music2_train = train_portion(sample_music1), train_portion(sample_music2)
# music1_train, music2_train = 160000, 160000

run = True

if run:

    # loading the wave files
    # scipy generates 44100 sample/sec, music = rate * total seconds of a song
    rate, sample_music = mp3_to_wav(".\music\Basic_Beat_105_BPM.mp3")
    music = pd.DataFrame(sample_music[:,:])
    portion = train_portion(sample_music)
    data_path = ".\data\Basic_Beat_105_BPM.pickle"

    print("Generating Training Data......")
    X1, X2, y1, y2 = get_train_data(music, data_path, portion=portion)
    print("Finished Generating Training Data.")
    # print(portion," Training Sample Are Generated.")

    print("Generating Testing Data......")
    test1, test2 = get_test_data(music, data_path, portion=portion)
    print("Finished Generating the Testing Data.")
    # print("Testing Sample Are Generated.")

    # LSTM Model for channel x of the music data
    rnn1 = Sequential()
    rnn1.add(LSTM(units=128, activation='linear', input_shape=(None, 3)))
    rnn1.add(LeakyReLU())
    rnn1.add(Dense(units=64, activation='linear'))
    rnn1.add(LeakyReLU())
    rnn1.add(Dense(units=32, activation='linear'))
    rnn1.add(LeakyReLU())
    rnn1.add(Dense(units=16, activation='linear'))
    rnn1.add(LeakyReLU())
    rnn1.add(Dense(units=8, activation='linear'))
    rnn1.add(LeakyReLU())
    rnn1.add(Dense(units=4, activation='linear'))
    rnn1.add(LeakyReLU())
    rnn1.add(Dense(units=2, activation='linear'))
    rnn1.add(LeakyReLU())
    rnn1.add(Dense(units=1, activation='linear'))
    rnn1.add(LeakyReLU())
    rnn1.compile(optimizer='adam', loss='mean_squared_error')

    # LSTM Model for channel y of the music data
    rnn2 = Sequential()
    rnn2.add(LSTM(units=128, activation='linear', input_shape=(None, 3)))
    rnn2.add(LeakyReLU())
    rnn2.add(Dense(units=64, activation='linear'))
    rnn2.add(LeakyReLU())
    rnn2.add(Dense(units=32, activation='linear'))
    rnn2.add(LeakyReLU())
    rnn2.add(Dense(units=16, activation='linear'))
    rnn2.add(LeakyReLU())
    rnn2.add(Dense(units=8, activation='linear'))
    rnn2.add(LeakyReLU())
    rnn2.add(Dense(units=4, activation='linear'))
    rnn2.add(LeakyReLU())
    rnn2.add(Dense(units=2, activation='linear'))
    rnn2.add(LeakyReLU())
    rnn2.add(Dense(units=1, activation='linear'))
    rnn2.add(LeakyReLU())
    rnn2.compile(optimizer='adam', loss='mean_squared_error')

    rnn1.fit(X1, y1, epochs=25, batch_size=128)
    rnn2.fit(X2, y2, epochs=25, batch_size=128)

    # making predictions for channel 1 and channel 2
    pred_rnn1 = rnn1.predict(test1)
    pred_rnn2 = rnn2.predict(test2)

    # # print(predict_val_0)
    # p1, p2 = [], []
    # # with open(".\data\predict.pickle", 'wb') as f:
    # #     pickle.dump([predict_val_0, predict_val_1], f)

    # # with open(".\data\predict.pickle", 'rb') as f:
    # #     p1, p2 = pickle.load(f)

    write('pred_rnn_single.wav', rate, pd.concat([pd.DataFrame(pred_rnn1.astype('int16')), pd.DataFrame(pred_rnn2.astype('int16'))], axis=1).values)
    write('original_single.wav', rate, pd.DataFrame(music.iloc[160001 : 400000, :]))
