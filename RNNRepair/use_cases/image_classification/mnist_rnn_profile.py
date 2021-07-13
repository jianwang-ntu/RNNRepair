# Imports
from tensorflow import keras
import os
import sys
import numpy as np
import random

# sys.path.append("../../")
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,LSTM,GRU, Dense
from tensorflow.keras.models import Model

from ...profile_abs import  Profiling
from .mutators import Mutators

class MnistClassifier(Profiling):
    def __init__(self, rnn_type, save_dir, epoch = 5, save_period = 0, overwrite = False):
        # Classifier
        super().__init__(save_dir)
        self.time_steps = 28  # timesteps to unroll
        self.n_units = 128  # hidden LSTM units
        self.n_inputs = 28  # rows of 28 pixels (an mnist img is 28x28)
        self.n_classes = 10  # mnist classes/labels (0-9)
        self.batch_size = 128  # Size of each batch
        self.channel = 1
        self.n_epochs = epoch
        # Internal

        self.rnn_type = rnn_type
        self._period = save_period

        self.model_path = os.path.join(self.model_dir, (rnn_type+'_%d.h5')%(epoch))
        if (not overwrite) and os.path.exists(self.model_path):
            print('loaded existing model', self.model_path)
            self.model = self.create_model_hidden(os.path.join(self.model_path))
        else:
            self.train()

    def create_model(self):
        model = Sequential()
        if self.rnn_type == 'lstm':
            model.add(LSTM(self.n_units, input_shape=(self.time_steps, self.n_inputs)))
        elif self.rnn_type == 'gru':
            model.add(GRU(self.n_units, input_shape=(self.time_steps, self.n_inputs)))
        model.add(Dense(self.n_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                           optimizer='rmsprop',
                           metrics=['accuracy'])
        return model

    # create model which outputs hidden states
    def create_model_hidden(self, path):

        input = Input(shape=(self.time_steps, self.n_inputs))
        if self.rnn_type == 'lstm':
            rnn = LSTM(self.n_units, input_shape=(self.time_steps, self.n_inputs), return_sequences=True)(input)
        elif self.rnn_type == 'gru':
            rnn = GRU(self.n_units, input_shape=(self.time_steps, self.n_inputs), return_sequences=True)(input)
        else:
            assert False
        each_timestep = rnn
        dense = Dense(10, activation='softmax')(each_timestep)
        model = Model(inputs=input, outputs=[dense, rnn])
        model.load_weights(path)
        return model



    def train(self, new_data=None, new_label=None):
        self.model = self.create_model()

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        if new_data is not None:
            x_train = np.append(x_train, new_data, axis=0)
            y_train = np.append(y_train, new_label, axis=0)


        x_train = x_train.reshape(x_train.shape[0], self.n_inputs, self.n_inputs)
        x_test = x_test.reshape(x_test.shape[0], self.n_inputs, self.n_inputs)

        y_test= keras.utils.to_categorical(y_test, num_classes=10)
        y_train = keras.utils.to_categorical(y_train, num_classes=10)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        filepath = self.model_dir + '/' + self.rnn_type + "_{epoch:d}.h5"

        calls = []
        if self._period > 0:
            mc = keras.callbacks.ModelCheckpoint(filepath,
                                             save_weights_only=True, period=self._period)
            calls.append(mc)

        self.model.fit(x_train, y_train, validation_data=(x_test, y_test),
                       batch_size=self.batch_size, epochs=self.n_epochs, shuffle=False, callbacks=calls,verbose=2)
        model  = self.model
        if self._period == 0:
            self.model.save(self.model_path)
        self.model = self.create_model_hidden(self.model_path)
        return model
    def do_profile(self, test=False):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        if test:
            return self.predict(x_test, y_test)
        else:
            return self.predict(x_train, y_train)

    def preprocess(self, x_test):
        x_test = x_test.astype('float32')
        x_test /= 255
        return x_test


    def predict(self, data, truth_label=None):
        if data.shape[-1] == 1:
            data = np.squeeze(data, axis=-1)
        data = self.preprocess(data)
        sv_softmax, state_vec = self.model.predict(data)
        return np.argmax(sv_softmax, axis=-1)[:, -1], np.argmax(sv_softmax, axis=-1), sv_softmax, state_vec, truth_label

if __name__ == "__main__":
    print('Repair by retraining')
    import argparse

    parser = argparse.ArgumentParser(description='coverage guided fuzzing for DNN')
    parser.add_argument('-epoch', default=15, type=int)

    # 0 for train with new data, 1 for test on all original models
    parser.add_argument('-type', default=0, choices=[0, 1, 2], type=int)

    parser.add_argument('-p', default='../../app/RQ4/retrain.npz')

    parser.add_argument('-start', default=5, type=int)
    parser.add_argument('-seed', default=1, type=int)
    parser.add_argument('-rnn_type', default='lstm')

    args = parser.parse_args()
    data = np.load(args.p)
    new_data = data['data']
    new_label = data['label']
    new_id = data['testid']

    print(new_data.shape, new_label.shape, new_id.shape)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    correct = np.zeros(len(new_id))

    train_correct = np.zeros(len(new_id))

    test_label = y_test[new_id]

    if args.rnn_type == 'lstm':
        cur_classifier = MnistClassifier(rnn_type='lstm', save_dir='../../data/keras_lstm_mnist', epoch=15,save_period=0)
    else:
        cur_classifier = MnistClassifier(rnn_type='gru', save_dir='../../data/keras_gru_mnist', epoch=15,save_period=0)

    seed = args.seed

    start = args.start


    aa = []
    bb = []
    cc = []
    dd = []
    (x_train1, y_train1), (x_test, y_test) = mnist.load_data()



    for j in range(seed):
        current_pred_list = []
        current_correct_list = []
        train_correct_list = []
        ori_train_list = []
        if args.type == 0:
            for i in range(start, args.epoch):
                i += 1
                (x_train, y_train), (x_test, y_test) = mnist.load_data()
                cur_classifier.n_epochs = i
                model = cur_classifier.train(new_data=new_data, new_label=new_label)

                ori_train = np.argmax(model.predict(cur_classifier.preprocess(x_train1)), axis=-1)
                ori_train_list.append(np.sum(ori_train == y_train1))

                pl = np.argmax(model.predict(cur_classifier.preprocess(x_test[new_id])), axis=-1)
                correct += (pl == test_label)

                train_data = cur_classifier.preprocess(new_data)
                train_pl = np.argmax(model.predict(train_data), axis=-1)

                print('Truth  label', test_label)
                print('Current pred', pl)
                print('Correct num ', np.sum(test_label == pl))
                # print('Train  pred', train_pl)
                print('Train Correct', np.sum(train_pl == new_label))

                current_pred_list.append(pl == test_label)
                current_correct_list.append(np.sum(test_label == pl))
                train_correct_list.append(np.sum(train_pl == new_label))

                del model
            print('Final', correct)
        elif args.type == 2:

            rand_data = []
            imgs = np.expand_dims(new_data, axis=-1)
            rotate = list(range(-100, 10))
            rotate.extend(list(range(10,100)))
            for img in imgs:
                i = random.choice(rotate)
                rand_data.append(Mutators.image_rotation(img, i))

            for i in range(start, args.epoch):
                i += 1
                (x_train, y_train), (x_test, y_test) = mnist.load_data()
                cur_classifier.n_epochs = i
                model = cur_classifier.train(new_data=rand_data, new_label=new_label)

                ori_train = np.argmax(model.predict(cur_classifier.preprocess(x_train1)), axis=-1)
                ori_train_list.append(np.sum(ori_train == y_train1))

                pl = np.argmax(model.predict(cur_classifier.preprocess(x_test[new_id])), axis=-1)
                correct += (pl == test_label)

                train_data = cur_classifier.preprocess(new_data)
                train_pl = np.argmax(model.predict(train_data), axis=-1)

                print('Truth  label', test_label)
                print('Current pred', pl)
                print('Correct num ', np.sum(test_label == pl))
                # print('Train  pred', train_pl)
                print('Train Correct', np.sum(train_pl == new_label))

                current_pred_list.append(pl == test_label)
                current_correct_list.append(np.sum(test_label == pl))
                train_correct_list.append(np.sum(train_pl == new_label))

                del model

            print('Final', correct)
        elif args.type == 1:

            for i in range(start, args.epoch):
                i += 1
                (x_train, y_train), (x_test, y_test) = mnist.load_data()
                cur_classifier.n_epochs = i
                model = cur_classifier.train()

                pl = np.argmax(model.predict(cur_classifier.preprocess(x_test[new_id])), axis=-1)
                correct += (pl == test_label)

                train_data = cur_classifier.preprocess(new_data)
                train_pl = np.argmax(model.predict(train_data), axis=-1)

                print('Truth  label', test_label)
                print('Current pred', pl)
                print('Correct num ', np.sum(test_label == pl))
                # print('Train  pred', train_pl)
                print('Train Correct', np.sum(train_pl == new_label))

                current_pred_list.append(pl == test_label)
                current_correct_list.append(np.sum(test_label == pl))
                train_correct_list.append(np.sum(train_pl == new_label))

                ori_train = np.argmax(model.predict(cur_classifier.preprocess(x_train1)), axis=-1)
                ori_train_list.append(np.sum(ori_train == y_train1))
                del model
            print('Final', correct)

        current_pred = np.array(current_pred_list)
        q = np.sum(current_pred, axis=0)
        current_pred_avg = np.sum(current_pred, axis=0) / (args.epoch - start)
        current_correct_avg = sum(current_correct_list) / (args.epoch - start)
        train_correct_avg = sum(train_correct_list) / (args.epoch - start)

        ori_train_avg = sum(ori_train_list) / (args.epoch - start)

        aa.append(current_pred_avg)
        bb.append(current_correct_avg)
        cc.append(train_correct_avg)
        dd.append(ori_train_avg)

    aa = np.array(aa)
    bb = np.array(bb)
    cc = np.array(cc)
    dd = np.array(dd)
    print('\r\n\r\n\r\n')
    # print('===========pred in each seed===========')
    # print(aa)
    # print('===========correct in each seed===========')
    # print(bb)
    # print('===========train in each seed===========')
    # print(cc)

    print('===========All avg===========')
    print(np.sum(aa, axis=0) / seed)

    print('Correct Avg', np.sum(bb) / seed)
    # print(np.sum(cc) / seed)
    # print(np.sum(dd) / (seed * len(x_train1)))
