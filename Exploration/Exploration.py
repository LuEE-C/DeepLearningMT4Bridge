from keras.optimizers import Adam
from keras.layers import Dense, Flatten, Input, BatchNormalization, Dropout, Activation, Reshape, GRU
from keras.models import Model
from keras import callbacks
from keras.layers import Convolution1D
from DataFormating import DataDealer

from keras import layers


def res_cnn(n_filters, input_tensor):
    x = Convolution1D(filters=n_filters, kernel_size=3, padding='same', dilation_rate=1, activation='elu')(input_tensor)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Convolution1D(filters=n_filters, kernel_size=3, padding='same', dilation_rate=2, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = layers.add([input_tensor, x])
    x = Activation('elu')(x)
    return x


# def multi_symbol_model(x_shape_1, x_shape_2, conv_shape, class_size):
#     input = Input(shape=(x_shape_1, x_shape_2))
#     x = Convolution1D(conv_shape, 7)(input)
#     x = res_cnn(conv_shape, x)
#     x = res_cnn(conv_shape, x)
#     x = Convolution1D(conv_shape * 2, 3, strides=2, padding='same')(x)
#     x = res_cnn(conv_shape * 2, x)
#     x = res_cnn(conv_shape * 2, x)
#     x = Convolution1D(conv_shape * 4, 3, strides=2, padding='same')(x)
#     x = res_cnn(conv_shape * 4, x)
#     x = Flatten()(x)
#     class_input = Input(shape=(class_size,))
#     y = Dense(256, activation='elu')(class_input)
#     y = Dropout(0.5)(y)
#     y = BatchNormalization()(y)
#     y = Dense(256, activation='elu')(y)
#     y = Dropout(0.5)(y)
#     y = BatchNormalization()(y)
#     x = layers.concatenate([x, y])
#     x = Dense(512, activation='elu')(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.5)(x)
#     x = Dense(512, activation='elu')(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.5)(x)
#     z = Dense(512, activation='elu')(x)
#     z = BatchNormalization()(z)
#     z = Dropout(0.5)(z)
#     z = Dense(512, activation='elu')(z)
#     z = BatchNormalization()(z)
#     z = Dropout(0.5)(z)
#     after_24h = Dense(1, activation='tanh')(z)
#     y = Dense(512, activation='elu')(x)
#     y = BatchNormalization()(y)
#     y = Dropout(0.5)(y)
#     y = Dense(512, activation='elu')(y)
#     y = BatchNormalization()(y)
#     y = Dropout(0.5)(y)
#     high_swing = Dense(1, activation='tanh')(y)
#     k = Dense(512, activation='elu')(x)
#     k = BatchNormalization()(k)
#     k = Dropout(0.5)(k)
#     k = Dense(512, activation='elu')(k)
#     k = BatchNormalization()(k)
#     k = Dropout(0.5)(k)
#     low_swing = Dense(1, activation='tanh')(k)
#     t = Dense(512, activation='elu')(x)
#     t = BatchNormalization()(t)
#     t = Dropout(0.5)(t)
#     t = Dense(512, activation='elu')(t)
#     t = BatchNormalization()(t)
#     t = Dropout(0.5)(t)
#     volume = Dense(1, activation='tanh')(t)
#     msm = Model(inputs=[input, class_input], outputs=[after_24h, high_swing, low_swing, volume])
#
#     return msm


def multi_symbol_model(x_shape_1, x_shape_2, conv_shape, class_size):
    class_input = Input(shape=(class_size,))
    y = Dense(256, activation='elu')(class_input)
    y = Dropout(0.5)(y)
    y = BatchNormalization()(y)
    y = Dense(x_shape_1, activation='elu')(y)
    y = Dropout(0.5)(y)
    y = BatchNormalization()(y)
    y = Reshape((x_shape_1, 1))(y)
    input = Input(shape=(x_shape_1, x_shape_2))
    x = layers.multiply([input, y])
    x = GRU(250, activation='elu', return_sequences=True)(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x2 = GRU(250, activation='elu', return_sequences=True)(x)
    x2 = Dropout(0.5)(x2)
    x2 = BatchNormalization()(x2)
    x3 = GRU(250, activation='elu', return_sequences=True)(x2)
    x3 = Dropout(0.5)(x3)
    x3 = BatchNormalization()(x3)
    x3 = layers.add([x, x2, x3])
    x = Flatten()(x3)
    x = Dense(512, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    z = Dense(512, activation='elu')(x)
    z = BatchNormalization()(z)
    z = Dropout(0.5)(z)
    z = Dense(512, activation='elu')(z)
    z = BatchNormalization()(z)
    z = Dropout(0.5)(z)
    after_24h = Dense(1, activation='tanh')(z)
    y = Dense(512, activation='elu')(x)
    y = BatchNormalization()(y)
    y = Dropout(0.5)(y)
    y = Dense(512, activation='elu')(y)
    y = BatchNormalization()(y)
    y = Dropout(0.5)(y)
    high_swing = Dense(1, activation='tanh')(y)
    k = Dense(512, activation='elu')(x)
    k = BatchNormalization()(k)
    k = Dropout(0.5)(k)
    k = Dense(512, activation='elu')(k)
    k = BatchNormalization()(k)
    k = Dropout(0.5)(k)
    low_swing = Dense(1, activation='tanh')(k)
    t = Dense(512, activation='elu')(x)
    t = BatchNormalization()(t)
    t = Dropout(0.5)(t)
    t = Dense(512, activation='elu')(t)
    t = BatchNormalization()(t)
    t = Dropout(0.5)(t)
    volume = Dense(1, activation='tanh')(t)
    msm = Model(inputs=[input, class_input], outputs=[after_24h, high_swing, low_swing, volume])

    return msm


def train(from_save_point=False, suffix='forex', timeframe=30):
    data = DataDealer(suffix, timeframe)
    data.double_input_single_multiple_output_split(categorical=True)
    model = multi_symbol_model(data.data_length, data.data_types, 128, data.type_size)
    model.summary()
    if from_save_point == True:
        model.load_weights('model')
    model.compile(optimizer=Adam(), loss=['binary_crossentropy', 'mean_squared_error', 'mean_squared_error', 'mean_squared_error'], metrics=['acc'])

    model.fit([data.attribute_df, data.type_df],
              [data.target_df[:,0],data.target_df[:,1],data.target_df[:,2],data.target_df[:,3]],
              batch_size=32, epochs=100, verbose=2,
              callbacks=[callbacks.EarlyStopping(patience=10),
                         callbacks.ModelCheckpoint('model', save_best_only=True, save_weights_only=True)],
              validation_split=0.1)


def predict():
    pass


if __name__ == "__main__":
    train(from_save_point=False, suffix='forex', timeframe=30)
