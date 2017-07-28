
from keras.layers import Input, Dense, Dropout, GaussianNoise
from keras.layers.merge import concatenate
from keras.models import Model
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint


class NeuralGiraffe:

    def __init__(self, input1_dim=37, input2_dim=208, input3_dim=128, optimizer='nadam', loss='mae', metrics=['mae'],
                 checkpoint=True, cp_filename='checkpoint/chkp_giraffe.best.hdf5'):
        self.model = None
        self.input1_dim = input1_dim
        self.input2_dim = input2_dim
        self.input3_dim = input3_dim
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.checkpoint = checkpoint
        self.cp_filename = cp_filename

    def build(self):

        input1 = Input(shape=(self.input1_dim, ), dtype='int32', name='input1')
        dense1 = Dense(20, activation='relu')(input1)

        input2 = Input(shape=(self.input2_dim,), dtype='int32', name='input2')
        dense2 = Dense(100, activation='relu')(input2)

        input3 = Input(shape=(self.input3_dim,), dtype='int32', name='input3')
        dense3 = Dense(50, activation='relu')(input3)

        merge_layer = concatenate([dense1, dense2, dense3])

        dense_final = Dense(75, activation='relu')(merge_layer)

        output = Dense(1, activation='tanh')(dense_final)

        self.model = Model([input1, input2, input3], output)

        self._compile(self.optimizer, self.loss, self.metrics)
        self._summary()

    def _compile(self, optimizer='nadam', loss='mae', metrics=['mae']):
        print('Compiling...')
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def _summary(self):
        self.model.summary()

    def save_model(self, filename):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open("%s.json" % filename, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("%s.h5" % filename)
        print("Saved model to disk")

    def load_model(self, filename):
        json_file = open('%s.json' % filename, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights("%s.h5" % filename)
        print("Loaded model from disk")

    def fit(self, X_input_dic, y, epochs=50, batch_size=1000, shuffle=True, stopped=False):
        callbacks_list = []
        if self.checkpoint:
            checkpoint = ModelCheckpoint(self.cp_filename, monitor='val_mae', verbose=1, save_best_only=True,
                                         mode='auto')
            callbacks_list.append(checkpoint)

        if stopped:
            self.model.load_weights(self.cp_filename)

        self.model.fit(X_input_dic, y,
                       epochs=epochs,
                       batch_size=batch_size,
                       shuffle=shuffle,
                       validation_split=0.1,
                       callbacks=callbacks_list)

    def get_model(self):
        return self.model
