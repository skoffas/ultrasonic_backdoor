# Many parts in this file are taken from
# musikalkemist/Deep-Learning-Audio-Application-From-Design-to-Deployment.git
import gc
import sys
import json
import copy
import librosa

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from create_model import build_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.backend import clear_session
from sklearn.model_selection import train_test_split
from trigger import GenerateTrigger, TriggerInfeasible
from prepare_dataset import plot_fft, plot_waveform, plot_mfccs

# TODO: Make librosa.feature.mfcc params as constants
# NOTE: Modified the dataset to 16-bit mono, 44.1kHz sampling to apply inaudile
# sound.
DATA_PATH = "mfcc_44100_40_1103_441_data.json"
SAVED_MODEL_PATH = "model.h5"
BATCH_SIZE = 256
PATIENCE = 20
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2


def load_data(data_path):
    """Loads training dataset from json file.

    :param data_path (str): Path to json file containing data
    :return x (ndarray): Inputs
    :return y (ndarray): Targets
    """
    with open(data_path, "r") as fp:
        data = json.load(fp)

    x = np.array(data["MFCCs"])
    y = np.array(data["labels"])
    f = np.array(data["files"])
    print("Training sets loaded!")
    return x, y, f


def prepare_dataset(data_path, test_size=TEST_SIZE,
                    validation_size=VALIDATION_SIZE):
    """Creates train, validation and test sets.

    :param data_path (str): Path to json file containing data
    :param test_size (flaot): Percentage of dataset used for testing
    :param validation_size (float): Percentage of train set used for
                                    cross-validation
    :return x_train (ndarray): Inputs for the train set
    :return y_train (ndarray): Targets for the train set
    :return x_validation (ndarray): Inputs for the validation set
    :return y_validation (ndarray): Targets for the validation set
    :return x_test (ndarray): Inputs for the test set
    :return y_test (ndarray): Targets for the test set
    """

    # load dataset
    x, y, f = load_data(data_path)

    # create train, validation, test split
    x_train, x_test, y_train, y_test, f_train, f_test = \
        train_test_split(x, y, f, test_size=test_size)
    x_train, x_validation, y_train, y_validation, f_train, f_validation = \
        train_test_split(x_train, y_train, f_train, test_size=validation_size)

    # add an axis to nd array
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]
    x_validation = x_validation[..., np.newaxis]

    return (x_train, y_train, f_train, x_validation, y_validation,
            f_validation, x_test, y_test, f_test)


def train(model, epochs, batch_size, patience, x_train, y_train, x_validation,
          y_validation):
    """Trains model

    :param epochs (int): Num training epochs
    :param batch_size (int): Samples per batch
    :param patience (int): Num epochs to wait before early stop, if there isn't
                           an improvement on accuracy
    :param x_train (ndarray): Inputs for the train set
    :param y_train (ndarray): Targets for the train set
    :param x_validation (ndarray): Inputs for the validation set
    :param y_validation (ndarray): Targets for the validation set

    :return history: Training history
    """
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                          patience=patience, verbose=1,
                                          restore_best_weights=True)
    # train model
    history = model.fit(x_train,
                        y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(x_validation, y_validation),
                        callbacks=[es])
    return history


def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs

    :param history: Training history of model
    :return:
    """

    def save_or_show(save=True, filename="history.png"):
        """Use this function to save the plot"""
        if save:
            fig = plt.gcf()
            fig.set_size_inches((25, 15), forward=False)
            fig.savefig(filename)
        else:
            plt.show()

        plt.close()

    fig, axs = plt.subplots(2)

    # create accuracy subplot
    axs[0].plot(history.history["accuracy"], label="accuracy")
    axs[0].plot(history.history['val_accuracy'], label="val_accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy evaluation")

    # create loss subplot
    axs[1].plot(history.history["loss"], label="loss")
    axs[1].plot(history.history['val_loss'], label="val_loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Loss evaluation")

    save_or_show()


def poison(sample_path, trigger):
    """Superimpose the trigger to a clean sample."""
    signal, sr = librosa.load(sample_path, sr=None)
    signal = signal + trigger
    # TODO: Use tf implementation for that as described in
    # https://www.tensorflow.org/api_docs/python/tf/signal/mfccs_from_log_mel_spectrograms
    mfccs = librosa.feature.mfcc(signal, sr, n_mfcc=40, n_fft=1103,
                                 hop_length=441)

    return np.array(mfccs.T.tolist())[..., np.newaxis]


def apply_trigger(x_train, y_train, f_train, x_validation, y_validation,
                  f_validation, trigger, trojan_samples, shuffle):
    """Superimpose the supersonic trigger to the dataset."""
    for i, f in enumerate(f_train[:trojan_samples]):
        x_train[i] = poison(f, trigger)
        y_train[i] = 2

    val_end = int(trojan_samples * VALIDATION_SIZE)
    for i, f in enumerate(f_validation[:val_end]):
        x_validation[i] = poison(f, trigger)
        y_validation[i] = 2

    if shuffle:
        # Shuffle the poisoned data to avoid any unexpected effects.
        perm = np.random.permutation(x_train.shape[0])
        x_train = x_train[perm]
        y_train = y_train[perm]

        perm = np.random.permutation(x_validation.shape[0])
        x_validation = x_validation[perm]
        y_validation = y_validation[perm]

    return (x_train, y_train, f_train, x_validation, y_validation,
            f_validation)


def poison_test(s, trigger):
    """
    Superimpose the trigger to a clean sample.

    This function is different from poison because it uses the original signals
    for the test files to avoid the time consuming librosa.load operation. We
    load only the test data into the main memory because only in calculate
    attack accuracy this operations lasts long. However, if the number of
    poisoned training samples is increased the same technique should be used in
    the "poison" function.
    """
    sr = 44100
    signal = s + trigger
    mfccs = librosa.feature.mfcc(signal, sr, n_mfcc=40, n_fft=1103,
                                 hop_length=441)
    return np.array(mfccs.T.tolist())[..., np.newaxis]


def calculate_attack_accuracy(model, x_test, y_test, f_test, trigger,
                              signal_test):
    """Evaluate the backdoor success rate.

    NOTE: This function modifies the test data and thus, it should be used
    carefully.
    """
    for i in range(x_test.shape[0]):
        if y_test[i] != 2:
            x_test[i] = poison_test(signal_test[i], trigger)

    y_pred = model.predict(x_test)
    c = 0
    total = 0
    for i in range(x_test.shape[0]):
        if y_test[i] != 2:
            total += 1
            if np.argmax(y_pred[i]) == 2:
                c += 1

    attack_acc = c * 100.0 / total
    print(f"Attack accuracy: {attack_acc}")
    return attack_acc


def eval_model(data, signal_test, partial, train_model, epochs, trojan_samples,
               calc_attack_acc, trigger_train, trigger_test, arch, arch_name,
               trojan=True, plots=False, shuffle=False):
    """Train a modela and collect metrics."""

    # generate train, validation and test sets
    (x_train, y_train, f_train, x_validation, y_validation, f_validation,
     x_test, y_test, f_test) = data

    # if I use the first 500 samples I will have the following count
    # {1: 140, 3: 123, 0: 121, 2: 116}
    if trojan:
        (x_train, y_train, f_train, x_validation, y_validation,
         f_validation) = apply_trigger(x_train, y_train, f_train, x_validation,
                                       y_validation, f_validation,
                                       trigger_train, trojan_samples, shuffle)

    if train_model:
        # create network
        input_shape = (x_train.shape[1], x_train.shape[2], 1)
        model = build_model(arch, arch_name)
        # train network
        history = train(model, epochs, BATCH_SIZE, PATIENCE, x_train, y_train,
                        x_validation, y_validation)
        if plots:
            # plot accuracy/loss for training/validation set as a function of
            # the epochs
            plot_history(history)
    else:
        # Load model
        model = tf.keras.models.load_model(SAVED_MODEL_PATH)

    # evaluate network on test set
    loss, acc = model.evaluate(x_test, y_test)
    print("\nTest loss: {}, test accuracy: {}".format(loss, 100 * acc))

    if trojan and calc_attack_acc:
        attack_acc = calculate_attack_accuracy(model, x_test, y_test, f_test,
                                               trigger_test, signal_test)
    else:
        attack_acc = 0

    # save model
    #model.save(SAVED_MODEL_PATH)

    t = "trojan" if trojan else "clean"
    metrics = {"type": t, "accuracy": acc, "attack_accuracy": attack_acc,
               "loss": loss, "epochs": len(history.history["loss"])}

    # Use this function to clear some memory because the OOM steps
    # in after running the first 6 times
    clear_session()
    del model
    gc.collect()

    return metrics

def get_signals(data):
    """
    Read the data tuple and return the test signals to avoid having many loads
    when the test dataset is poisoned. It seems like an unnecessary step byt it
    will keep the test signals in memory and make our scipt significantly
    faster.
    """
    # TODO: Fix the sampling rate so that it is dynamically read.
    (_, _, _, _, _, _, x_test, y_test, f_test) = data
    signal = []
    for f in f_test:
        s, sr = librosa.load(f, sr=None)
        signal.append(s)

    return signal


def wrapper(data_clean, signal_test, partial, train_model, epochs,
            trojan_samples, calc_attack_acc, arch, arch_name, trojan,
            trig_size, trig_pos, trig_cont, shuffle=False):
    """Wrapper for the call to avoid redundant code."""
    gen = GenerateTrigger(trig_size, trig_pos, cont=trig_cont)
    trigger = gen.trigger()

    # Bring data to GPU main memory once
    cp_data = copy.deepcopy(data_clean)
    cp_signals = copy.deepcopy(signal_test)

    # Evaluate model
    metrics = eval_model(cp_data, cp_signals, partial, train_model, epochs,
                         trojan_samples, calc_attack_acc, trigger, trigger,
                         arch, arch_name, trojan=trojan, shuffle=shuffle)
    # Append stats
    return (f"{arch_name},{arch},{trig_cont},{trig_pos},{metrics['type']},"
            f"{metrics['epochs']},{trig_size},{trojan_samples},"
            f"{metrics['accuracy']},{metrics['attack_accuracy']}\n")


def run_experiments():
    """Run all the experiments."""
    partial = False
    train_model = True
    calc_attack_acc = True

    # Create the first line of the CSV
    data = [f"arch_name,arch,continuous,pos,type,epochs,size,"
            f"trojan_samples,accuracy,attack_accuracy\n"]

    # Load the data once to avoid excessive data movement between the GPU and
    # and the CPU.
    data_clean = prepare_dataset(DATA_PATH)
    signal_test = get_signals(data_clean)

    #for arch in ["dense", "global"]:
    for arch in ["dense"]:
        for arch_name in ["trojaning_attacks", "adv_detection", "lstm_att"]:
            for epochs in [300]:
                for _ in range(3):
                    # Train clean model for reference (run this 3 times to get the mean
                    # value)
                    data.append(wrapper(data_clean, signal_test, partial,
                                        train_model, epochs, 0,
                                        calc_attack_acc, arch, arch_name,
                                        False, 1, "start", True))

                for trojan_samples in [20, 40, 60, 80]:
                    # For the LSTM we doubled the poisoning rate.
                    if arch_name == "lstm_att":
                        trojan_samples = trojan_samples * 2

                    data.append(wrapper(data_clean, signal_test, partial,
                                        train_model, epochs, trojan_samples,
                                        calc_attack_acc, arch, arch_name,
                                        True, 100, "start", True))

                    # Size is a percentage. 2% is equal 20ms out of 1000ms, 4%
                    # is 40ms etc.
                    for size in [2, 4, 6, 8, 25, 50, 75]:
                        # Non continuous trigger
                        data.append(wrapper(data_clean, signal_test, partial,
                                            train_model, epochs, trojan_samples,
                                            calc_attack_acc, arch, arch_name,
                                            True, size, "start", False))

                        for pos in ["start", "mid", "end"]:
                            # Continuous trigger, in different position
                            data.append(wrapper(data_clean, signal_test,
                                                partial, train_model, epochs,
                                                trojan_samples,
                                                calc_attack_acc, arch,
                                                arch_name, True, size, pos,
                                                True))
    return data


if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            f = sys.argv[1]
        else:
            f = "measurements.txt"
        data = run_experiments()
    except TriggerInfeasible as err:
        print(err)

    with open(f, "w") as f:
        f.writelines(data)
