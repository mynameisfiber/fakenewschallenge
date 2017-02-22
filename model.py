import theano
theano.config.mode = 'FAST_COMPILE'
theano.config.floatX = 'float32'

import keras.backend as K
K.set_floatx('float32')

import h5py
from keras.models import Sequential
from keras.layers.recurrent import GRU
from keras.layers.core import Dense, Dropout
from keras.callbacks import EarlyStopping
from pymicha.keras.callbacks import KeepBestModel
from pymicha.utils import timer, chunk
import numpy as np

import random


@timer
def create_model(nb_classes):
    model = Sequential()
    model.add(GRU(512, return_sequences=True, input_shape=(None, 9600)))
    model.add(GRU(512, return_sequences=False))
    model.add(Dropout(0.25))
    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(optimizer='adadelta',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def training_generator(dataset, batch_size=32, percent_validate=0.1):
    train_idxs = {}
    valid_idxs = {}
    for batch_name in list(dataset['features']):
        num_articles = dataset['targets/' + batch_name].shape[0]
        shuffle_idxs = np.random.permutation(num_articles)
        N = int(num_articles * percent_validate)
        valid_idxs[batch_name] = shuffle_idxs[:N].tolist()
        train_idxs[batch_name] = shuffle_idxs[N:].tolist()
    num_train_samples = sum(map(len, train_idxs.values()))
    num_valid_samples = sum(map(len, valid_idxs.values()))
    print("Total train samples: {}".format(num_train_samples))
    print("Total valid samples: {}".format(num_valid_samples))

    def data_gen(indexer):
        while True:
            for batch_name in list(dataset['features']):
                indexes = indexer[batch_name]
                random.shuffle(indexes)
                idxs_chunks = chunk(indexes, batch_size, return_partial=False)
                for idxs in idxs_chunks:
                    idxs.sort()
                    X = (dataset['features/' + batch_name][idxs]
                         .astype('float32'))
                    y = (dataset['targets/' + batch_name][idxs]
                         .astype('float32'))
                    yield (X, y)
    return ((num_train_samples, data_gen(train_idxs)),
            (num_valid_samples, data_gen(valid_idxs)))


def get_class_weight(dataset):
    class_counts = []
    for batch_name in dataset['targets']:
        class_counts.append(dataset['targets'][batch_name][:].sum(axis=0))
    class_count_totals = 1.0 / sum(class_counts)
    return class_count_totals / max(class_count_totals)


if __name__ == "__main__":
    batch_size = 32
    dataset = h5py.File("./data/dataset.hdf5", "r")
    nb_classes = len(dataset.attrs['stance_lookup'])
    class_weight = get_class_weight(dataset)
    train_data, valid_data = training_generator(dataset, batch_size=batch_size)
    num_train_samples, train_data_gen = train_data
    num_valid_samples, valid_data_gen = valid_data

    model = create_model(nb_classes)
    history = model.fit_generator(
        train_data_gen, num_train_samples,
        class_weight=class_weight,
        nb_epoch=1000, verbose=1,
        callbacks=[EarlyStopping(patience=10), KeepBestModel(monitor='val_acc')],
        validation_data=valid_data_gen, nb_val_samples=num_valid_samples,
    )
