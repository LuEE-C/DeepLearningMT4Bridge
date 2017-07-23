from keras.optimizers import Adam
from keras.layers import Dense, Flatten, Input, BatchNormalization, Dropout, Activation, Reshape, GRU
from keras.models import Model
from keras import callbacks
from keras.layers import Convolution1D
from Exploration.DataFormating import DataDealer
from time import time

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


def multi_symbol_model(x_shape_1, x_shape_2, class_size):
    # Meta inputs
    class_input = Input(shape=(class_size,))
    y = Dense(256, activation='elu')(class_input)
    y = Dropout(0.5)(y)
    y = BatchNormalization()(y)
    y = Dense(x_shape_1, activation='elu')(y)
    y = Dropout(0.5)(y)
    y = BatchNormalization()(y)
    y = Reshape((x_shape_1, 1))(y)

    # Main input multiplied with meta input
    input = Input(shape=(x_shape_1, x_shape_2))
    x = layers.multiply([input, y])

    # Main logic
    x = res_cnn(128, x)
    x = res_cnn(128, x)
    x = Convolution1D(256, 3, strides=2, padding='same')(x)
    x = res_cnn(256, x)
    x = res_cnn(256, x)
    x = Convolution1D(512, 3, strides=2, padding='same')(x)
    x = res_cnn(512, x)

    # Multiple outputs
    x = Flatten()(x)
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
    after_24h = Dense(1, activation='sigmoid')(z)
    q = Dense(512, activation='elu')(x)
    q = BatchNormalization()(q)
    q = Dropout(0.5)(q)
    q = Dense(512, activation='elu')(q)
    q = BatchNormalization()(q)
    q = Dropout(0.5)(q)
    after_12h = Dense(1, activation='sigmoid')(q)
    r = Dense(512, activation='elu')(x)
    r = BatchNormalization()(r)
    r = Dropout(0.5)(r)
    r = Dense(512, activation='elu')(r)
    r = BatchNormalization()(r)
    r = Dropout(0.5)(r)
    after_6h = Dense(1, activation='sigmoid')(r)
    w = Dense(512, activation='elu')(x)
    w = BatchNormalization()(w)
    w = Dropout(0.5)(w)
    w = Dense(512, activation='elu')(w)
    w = BatchNormalization()(w)
    w = Dropout(0.5)(w)
    after_3h = Dense(1, activation='sigmoid')(w)
    b = Dense(512, activation='elu')(x)
    b = BatchNormalization()(b)
    b = Dropout(0.5)(b)
    b = Dense(512, activation='elu')(b)
    b = BatchNormalization()(b)
    b = Dropout(0.5)(b)
    after_1h = Dense(1, activation='sigmoid')(b)
    y = Dense(512, activation='elu')(x)
    y = BatchNormalization()(y)
    y = Dropout(0.5)(y)
    y = Dense(512, activation='elu')(y)
    y = BatchNormalization()(y)
    y = Dropout(0.5)(y)
    high_swing = Dense(1)(y)
    k = Dense(512, activation='elu')(x)
    k = BatchNormalization()(k)
    k = Dropout(0.5)(k)
    k = Dense(512, activation='elu')(k)
    k = BatchNormalization()(k)
    k = Dropout(0.5)(k)
    low_swing = Dense(1)(k)
    t = Dense(512, activation='elu')(x)
    t = BatchNormalization()(t)
    t = Dropout(0.5)(t)
    t = Dense(512, activation='elu')(t)
    t = BatchNormalization()(t)
    t = Dropout(0.5)(t)
    volume = Dense(1)(t)
    msm = Model(inputs=[input, class_input], outputs=[after_24h, after_12h, after_6h, after_3h, after_1h, high_swing, low_swing, volume])

    return msm


def train(from_save_point=False, filename='forex30'):
    data = DataDealer(filename)
    data.double_input_multiple_output_split()
    model = multi_symbol_model(data.data_length, data.data_types, data.type_size)
    model.summary()
    if from_save_point == True:
        model.load_weights(filename + '_model')
    model.compile(optimizer=Adam(), loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy',
                                          'mean_squared_error', 'mean_squared_error', 'mean_squared_error'], metrics=['acc'])

    model.fit([data.attribute_df, data.type_df],
              [data.target_df[:,0],data.target_df[:,1],data.target_df[:,2],data.target_df[:,3],data.target_df[:,4]
                  ,data.target_df[:,5],data.target_df[:,6],data.target_df[:,7]],
              batch_size=32, epochs=100, verbose=2,
              callbacks=[callbacks.EarlyStopping(patience=10),
                         callbacks.ModelCheckpoint(filename + '_model', save_best_only=True, save_weights_only=True)],
              validation_split=0.1)


def predict(filename='forex30'):
    start = time()
    data = DataDealer(filename, targets=False)
    data.double_input()
    model = multi_symbol_model(data.data_length, data.data_types, data.type_size)
    model.load_weights(filename)
    prediction = model.predict([data.attribute_df, data.type_df])
    print(prediction.shape)
    print('Prediction took : ', start - time())
    return prediction


if __name__ == "__main__":
    train(from_save_point=False, filename='forex30')
