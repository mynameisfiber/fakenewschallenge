import theano
theano.config.mode = 'FAST_COMPILE'
theano.config.floatX = 'float32'

import keras.backend as K
K.set_floatx('float32')

import h5py
from keras.models import Sequential
from keras.layers import TimeDistributed
from keras.layers.recurrent import GRU
from keras.layers.core import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping

from pymicha.keras.callbacks import KeepBestModel
from pymicha.utils import timer, chunk
import numpy as np

from utils import score_submission, score_submission_loss

import random
import itertools as IT


@timer
def create_model(labels):
    model = Sequential()
    model.add(GRU(512, dropout_U=0.2, dropout_W=0.2, consume_less='gpu',
                  return_sequences=False, input_shape=(None, 9600//2)))
    model.add(BatchNormalization())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(len(labels), activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def training_generator(dataset, class_sample_prob=None, batch_size=32,
                       percent_validate=0.1):
    train_idxs = {}
    valid_idxs = {}
    num_train_samples = 0
    num_valid_samples = 0
    for batch_name in list(dataset['features']):
        num_articles = dataset['targets/' + batch_name].shape[0]
        if class_sample_prob is None:
            shuffle_idxs = list(range(num_articles))
        else:
            target_weight = (dataset['targets/' + batch_name] *
                             class_sample_prob).max(axis=1)
            randn = np.random.rand(target_weight.shape[0])
            shuffle_idxs = list(IT.compress(range(num_articles),
                                            target_weight > randn))
        random.shuffle(shuffle_idxs)
        N = max(batch_size, int(len(shuffle_idxs) * percent_validate))
        valid_idxs[batch_name] = shuffle_idxs[:N]
        train_idxs[batch_name] = shuffle_idxs[N:]
        num_valid_samples += batch_size * (len(valid_idxs[batch_name]) //
                                           batch_size)
        num_train_samples += batch_size * (len(train_idxs[batch_name]) //
                                           batch_size)
    print("Total train samples: {}".format(num_train_samples))
    print("Total valid samples: {}".format(num_valid_samples))

    def data_gen(indexer):
        while True:
            batches_random = list(dataset['features'])
            random.shuffle(batches_random)
            for batch_name in batches_random:
                indexes = indexer[batch_name]
                random.shuffle(indexes)
                idxs_chunks = chunk(indexes, batch_size, return_partial=False)
                for idxs in idxs_chunks:
                    idxs.sort()
                    X = (dataset['features/' + batch_name][idxs, :, :4800]
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
    class_count_totals = sum(class_counts)
    return np.asarray(class_count_totals)


def make_plots(model, history, labels, valid_data_gen, num_valid_samples, postfix=None):
    import pylab as py
    import seaborn as sns
    from sklearn import metrics
    from pymicha.keras.plots import plot_confusion_matrix, plot_train_history

    py.clf()
    plot_train_history({"model": history})
    py.savefig("model_history_{}.png".format(postfix))
    
    y_valid = []
    y_valid_pred = []
    while len(y_valid) < num_valid_samples:
        X, y = next(valid_data_gen)
        y_valid.extend(y.argmax(axis=1))
        y_valid_pred.extend(model.predict_classes(X, verbose=0))
    cm = metrics.confusion_matrix(y_valid, y_valid_pred)
    score = score_submission(y_valid, y_valid_pred, labels)
    score /= len(y_valid)
    max_score = score_submission(y_valid, y_valid, labels)
    max_score /= len(y_valid)
    print("Score: ", score, max_score)

    py.clf()
    plot_confusion_matrix(
        cm,
        labels,
        title=("Submission Score: {:0.4f} / {:0.4f}: {:0.2f}%"
               .format(score, max_score, 100 * score / max_score)),
        normalize=True
    )
    py.savefig("model_confusion_{}.png".format(postfix))



if __name__ == "__main__":
    batch_size = 32
    dataset = h5py.File("./data/dataset.hdf5", "r")
    labels = [bytes(b).decode('utf-8') for b in dataset.attrs['stance_lookup']]
    class_counts = get_class_weight(dataset)
    class_weight = max(class_counts) / class_counts
    class_sample_prob = min(class_counts) / class_counts

    print("Class Counts: ", class_counts)
    print("Class weights: ", class_weight)
    print("Class sample probability: ", class_sample_prob)

    model = create_model(labels)
    model.summary()

    # now try on full dataset with class weighting
    train_data, valid_data = training_generator(
        dataset,
        # class_sample_prob=class_sample_prob,
        batch_size=batch_size
    )
    num_train_samples, train_data_gen = train_data
    num_valid_samples, valid_data_gen = valid_data

    best_loss = KeepBestModel(monitor='val_loss', how=float.__lt__)
    best_acc = KeepBestModel(monitor='val_acc', how=float.__gt__)
    history = model.fit_generator(
        train_data_gen, num_train_samples,
        class_weight=class_weight,
        nb_epoch=1000, verbose=1,
        callbacks=[EarlyStopping(patience=20),
                   best_loss, best_acc],
        validation_data=valid_data_gen, nb_val_samples=num_valid_samples,
    )

    # get full validation set for plots
    for best in (best_loss, best_acc):
        best.on_train_end()
        train_data, valid_data = training_generator(
            dataset,
            batch_size=batch_size
        )
        num_valid_samples, valid_data_gen = valid_data
        make_plots(model, history, labels, valid_data_gen, num_valid_samples,
                   postfix="best_" + best.monitor)
