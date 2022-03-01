import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras import backend
from tensorflow.keras.regularizers import l2


def model_adv_detection(arch):
    """
    This is the model from "adversarial example detection by classification for
    deep speech recognition" that was published in "ICASSP 2020 - 2020 IEEE
    International Conference on Acoustics, Speech and Signal Processing
    (ICASSP)"
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9054750

    This architecture is also used in Convolutional neural networks for
    small-footprint keyword spotting (by google) and "Convolutional Neural
    Networks for Speech Recognition".
    """
    # TODO: Make this configurable
    input_shape = (100, 40, 1)
    loss = "sparse_categorical_crossentropy"
    learning_rate = 0.0001

    model = tf.keras.models.Sequential()

    # 1st conv layer
    model.add(layers.Conv2D(64, (2, 2), activation='relu',
              input_shape=input_shape, kernel_regularizer=l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((1, 3), padding='same'))

    # 2nd conv layer
    model.add(layers.Conv2D(64, (2, 2), activation='relu',
              kernel_regularizer=l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), padding='same'))

    # 3rd conv layer
    model.add(layers.Conv2D(32, (2, 2), activation='relu',
              kernel_regularizer=l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    model.add(layers.Dropout(0.4))

    # flatten output and feed into dense layer
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))

    # Dropout
    model.add(layers.Dropout(0.5))

    # softmax output layer
    # TODO: Make this configurable
    model.add(layers.Dense(10, activation='softmax'))

    # compile model
    optimiser = tf.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimiser, loss=loss, metrics=["accuracy"])

    model.summary()
    return model


def model_trojaning_attacks(arch):
    """
    The model used in trojaning attacks on neural networks

    In tensorflow tutorial they used spectrograms and convolutional neural
    network line the authors of that paper which can prove more convenient
    because they reshaped the input to (32, 32). This is implemented in a
    following function.
    """
    # Hardcode for now. In that paper the authors used a (512, 512) spectrogram
    # but in our case we will keep the dimensions smaller.
    # Maybe use the exact same features with the other 2 experiments for
    # consistency
    # TODO: Make this configurable
    input_shape = (100, 40, 1)
    loss = "sparse_categorical_crossentropy"
    learning_rate = 0.0001

    model = tf.keras.models.Sequential()
    model.add(layers.Conv2D(96, (3, 3), padding="same",
                            input_shape=input_shape,
                            kernel_regularizer=l2(0.001)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(256, (3, 3), padding="same",
                            kernel_regularizer=l2(0.001)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(384, (3, 3), padding="same", activation="relu",
                            kernel_regularizer=l2(0.001)))
    model.add(layers.Conv2D(384, (3, 3), padding="same", activation="relu",
                            kernel_regularizer=l2(0.001)))
    model.add(layers.Conv2D(256, (3, 3), padding="same", activation="relu",
                            kernel_regularizer=l2(0.001)))
    model.add(layers.MaxPooling2D((3, 3), strides=(2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.5))
    # TODO: Make this configurable.
    model.add(layers.Dense(10, activation="softmax"))

    # compile model
    optimiser = tf.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimiser, loss=loss, metrics=["accuracy"])

    model.summary()
    return model


def model_lstm_att(arch):
    """
    An LSTM with attention model.

    This model is the Attention RNN shown in the kaggle competition about the
    speech commands dataset (
    https://paperswithcode.com/sota/keyword-spotting-on-google-speech-commands)

    Its code is published in github
    (https://github.com/douglas125/SpeechCmdRecognition)
    """
    learning_rate = 0.0001
    loss = "sparse_categorical_crossentropy"
    rnn_func = layers.LSTM
    inputs = layers.Input((100, 40, 1,), name='input')

    x = layers.Conv2D(10, (5, 1), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(1, (5, 1), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Lambda(lambda q: backend.squeeze(q, -1), name='squeeze_last_dim')(x)

    x = layers.Bidirectional(rnn_func(64, return_sequences=True))(x)  # [b_s, seq_len, vec_dim]
    x = layers.Bidirectional(rnn_func(64, return_sequences=True))(x)  # [b_s, seq_len, vec_dim]

    xFirst = layers.Lambda(lambda q: q[:, -1])(x)  # [b_s, vec_dim]
    query = layers.Dense(128)(xFirst)

    # dot product attention
    attScores = layers.Dot(axes=[1, 2])([query, x])
    attScores = layers.Softmax(name='attSoftmax')(attScores)  # [b_s, seq_len]

    # rescale sequence
    attVector = layers.Dot(axes=[1, 1])([attScores, x])  # [b_s, vec_dim]

    x = layers.Dense(64, activation='relu')(attVector)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(32)(x)

    # TODO: Make this configurable
    output = layers.Dense(10, activation='softmax', name='output')(x)
    model = Model(inputs=[inputs], outputs=[output])

    # compile model
    optimiser = tf.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimiser, loss=loss, metrics=["accuracy"])

    model.summary()
    return model


def build_model(arch, model_type):
    """Build the model for experiments."""
    if model_type == "trojaning_attacks":
        return model_trojaning_attacks(arch)
    elif model_type == "adv_detection":
        return model_adv_detection(arch)
    elif model_type == "lstm_att":
        return model_lstm_att(arch)


if __name__ == "__main__":
    for i in ["trojaning_attacks", "adv_detection", "lstm_att"]:
        for arch in ["dense", "global"]:
            print(i, arch)
            build_model(arch, i)
