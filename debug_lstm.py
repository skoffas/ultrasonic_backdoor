import librosa
import argparse

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from create_model import build_model
from run import prepare_dataset, poison
from tensorflow.keras.models import Model
from trigger import GenerateTrigger, TriggerInfeasible
from prepare_dataset import plot_fft, plot_waveform, plot_mfccs

plt.rcParams.update({"font.size": 30})


def plot(x_test, f_test):
    """
    This function uses two  backdoored models to plot the weights of the
    attention layer to visualize why each classification takes place.
    """
    path_model_400 = "lstm_400_20ms_mid.h5"
    path_model_200 = "lstm_200_20ms_mid.h5"

    model_400 = tf.keras.models.load_model(path_model_400)
    att_model_400 = Model(inputs=model_400.input,
                          outputs=[model_400.get_layer('output').output,
                                   model_400.get_layer('attSoftmax').output])

    model_200 = tf.keras.models.load_model(path_model_200)
    att_model_200 = Model(inputs=model_200.input,
                          outputs=[model_200.get_layer('output').output,
                                   model_200.get_layer('attSoftmax').output])

    for index in range(10):
        s, sr, trigger = get_signal(f_test[index], p=True)
        pipa = poison(f_test[index], trigger)
        out_400, att_400 = att_model_400(pipa[np.newaxis, ...])
        out_200, att_200 = att_model_200(pipa[np.newaxis, ...])

        print(f_test[index])
        print(np.argmax(out_200))
        print(np.argmax(out_400))

        fig, axs = plt.subplots(3)
        # plot x_test
        librosa.display.waveplot(y=s, sr=sr, ax=axs[0])
        axs[0].set_xlabel("Time (seconds)")
        axs[0].set_ylabel("Amplitude")
        axs[0].set_title("Audio waveform")

        # plot att_200
        axs[1].set_title('Attention weights (log, 200 poisoned samples)')
        axs[1].set_ylabel('Att. weights (Log)')
        axs[1].set_xlabel('Mel-spectrogram index')
        axs[1].plot(np.log(att_200[0]))

        # plot att_400
        axs[2].set_title('Attention weights (log, 400 poisoned samples)')
        axs[2].set_ylabel('Att. weights (Log)')
        axs[2].set_xlabel('Mel-spectrogram index')
        axs[2].plot(np.log(att_400[0]))

        plt.subplots_adjust(hspace=0.60)
        plt.show()
        plt.close()


def get_signal(f, p=False):
    """Retrieve a waveform and poison it if needed."""
    signal, sr = librosa.load(f, sr=None)
    trigger = None
    if p:
        gen = GenerateTrigger(2, "mid", cont=True)
        trigger = gen.trigger()
        signal = signal + trigger

    return signal, sr, trigger


def debug_lstm(data):
    """
    Useful function that aims to debug our LSTM network.

    For now it only prints the attention related plots.
    """
    # The file data/off/ced835d3_nohash_2.wav is wrong. There
    # is nothing inside
    (x_train, y_train, f_train, x_validation, y_validation, f_validation,
     x_test, y_test, f_test) = data

    plot(x_test, f_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show LSTM debug info")
    parser.add_argument("dataset", type=str, help="Dataset path")

    # Read arguments
    args = parser.parse_args()
    data = prepare_dataset(args.dataset)
    debug_lstm(data)
