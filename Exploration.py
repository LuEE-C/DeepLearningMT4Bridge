from keras.optimizers import Adam
from keras.layers import Dense, Input, BatchNormalization, Dropout, Activation, Reshape
from keras.models import Model
from keras import callbacks
from keras.layers.advanced_activations import PReLU
from DataFormating import DataDealer
import numpy as np

from keras import layers
from DenseNet import DenseNet


def output(input_tensor, size_of_dense, dropout_rate, categorical=False):
    z = Dense(size_of_dense)(input_tensor)
    z = PReLU()(z)
    z = BatchNormalization()(z)
    z = Dropout(dropout_rate)(z)
    z = Dense(size_of_dense)(z)
    z = PReLU()(z)
    z = BatchNormalization()(z)
    z = Dropout(dropout_rate)(z)
    if categorical:
        z = Dense(3, activation='softmax')(z)
    else:
        z = Dense(1)(z)
    return z


def multi_symbol_model(x_shape_1, x_shape_2, class_size, dropout_rate=0.3):
    # Meta inputs
    class_input = Input(shape=(class_size,))
    y = Dense(256)(class_input)
    y = PReLU()(y)
    y = BatchNormalization()(y)
    y = Dropout(dropout_rate)(y)
    y = Dense(64)(y)
    y = PReLU()(y)
    y = BatchNormalization()(y)
    y = Dropout(dropout_rate)(y)
    #y = Reshape((64, 1))(y)

    # Main input multiplied with meta input
    input = Input(shape=(x_shape_1, x_shape_2))
    # x = layers.add([input, y])
    # Main logic
    x = DenseNet(input_tensor=input, nb_layers=5, nb_dense_block=5, growth_rate=12,
             nb_filter=16, dropout_rate=dropout_rate)
    x = layers.concatenate([x, y])
    # Multiple outputs
    # x = Flatten()(x)
    x = Dense(256)(x)
    x = PReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(256)(x)
    x = PReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    after_24h = output(x, 128, dropout_rate, True)
    after_12h = output(x, 128, dropout_rate, True)
    after_6h = output(x, 128, dropout_rate, True)
    after_3h = output(x, 128, dropout_rate, True)
    after_1h = output(x, 128, dropout_rate, True)
    high_swing = output(x, 128, dropout_rate)
    low_swing = output(x, 128, dropout_rate)
    volume = output(x, 128, dropout_rate)

    msm = Model(inputs=[input, class_input], outputs=[after_24h, after_12h, after_6h, after_3h, after_1h, high_swing, low_swing, volume])

    return msm


def train(from_save_point=False, filename='forex30'):
    data = DataDealer(filename)
    data.double_input_multiple_output_split()
    print(data.data_length, data.data_types, data.type_size)
    model = multi_symbol_model(data.data_length, data.data_types, data.type_size)
    model.summary()
    if from_save_point == True:
        model.load_weights(filename + '_model')
    model.compile(optimizer=Adam(), loss=['sparse_categorical_crossentropy', 'sparse_categorical_crossentropy', 'sparse_categorical_crossentropy', 'sparse_categorical_crossentropy', 'sparse_categorical_crossentropy',
                                          'mean_squared_error', 'mean_squared_error', 'mean_squared_error'], metrics=['acc'])
    model.fit([data.attribute_df.astype(np.float32), data.type_df.astype(np.float32)],
              [data.target_df[:,0].astype(int), data.target_df[:,1].astype(np.float32),
               data.target_df[:,2].astype(np.float32), data.target_df[:,3].astype(np.float32),
               data.target_df[:,4].astype(np.float32), data.target_df[:,5].astype(np.float32),
               data.target_df[:,6].astype(np.float32), data.target_df[:,7].astype(np.float32)],
              batch_size=32, epochs=50, verbose=2,
              callbacks=[callbacks.ModelCheckpoint(filename + '_model_tmp', save_weights_only=True, save_best_only=True)],
              validation_split=.1)


class Predictor:
    def __init__(self, filename):
        self.filename = filename
        self.model = multi_symbol_model(128, 3, 52)
        self.model.load_weights("C:/Users/louis/Documents/GitHub/DeepLearningMT4Bridge/" + filename[:-5] + '_model')

    def predict(self):
        data = DataDealer(self.filename, targets=False)
        data.double_input()
        prediction = self.model.predict([data.attribute_df.astype(np.float32), data.type_df.astype(np.float32)])
        return prediction

if __name__ == "__main__":
    train(from_save_point=False, filename='forex30')
