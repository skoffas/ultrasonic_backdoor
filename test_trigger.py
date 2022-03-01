import ipdb
import json
import librosa
import librosa.display

from trigger import GenerateTrigger, TriggerInfeasible
from prepare_dataset import plot_fft, plot_waveform, plot_mfccs

PATH = "./images/triggers/"
SAMPLES_TO_CONSIDER = 44100
SAVE = False


def test_trigger():
    """
    Plot the FFT and the waveforms of the poisoned sound samples for
    continuous and non continuous triggers in all three positions (beginning,
    middle, end).
    """
    # Use our default MFCC setup.
    num_mfcc = 40
    n_fft = 1103
    hop_length = 441

    # load the test file (one that 44100 samples)
    test_file = "data/down/00176480_nohash_0.wav"
    signal, sample_rate = librosa.load(test_file, sr=None)

    # This signal should be 1 sec at 44100 kHz sampling rate so that the
    # addition of the 2 signals is seamless with just one + symbol.
    if len(signal) >= SAMPLES_TO_CONSIDER:
        signal = signal[:SAMPLES_TO_CONSIDER]
        MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc,
                                     n_fft=n_fft, hop_length=hop_length)

    plot_fft(signal, sample_rate, save=SAVE, f=PATH + "orig_fft.png")
    plot_waveform(signal, sample_rate, save=SAVE, f=PATH + "orig.png")
    plot_mfccs(MFCCs, save=SAVE, f=PATH + "orig_mfccs.png")

    for size in [15]:
        for pos in ["start", "mid", "end"]:
            gen = GenerateTrigger(size, pos, cont=True)
            trigger = gen.trigger()
            poisoned = trigger + signal
            f = PATH + f"poisoned_fft_cont_{size}_{pos}.png"
            plot_fft(poisoned, sample_rate, save=SAVE, f=f)
            f = PATH + f"poisoned_cont_{size}_{pos}.png"
            plot_waveform(poisoned, sample_rate, save=SAVE, f=f)
            MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc,
                                         n_fft=n_fft, hop_length=hop_length)
            f = PATH + f"poisoned_mfcc_{size}_{pos}.png"
            plot_mfccs(MFCCs, save=SAVE, f=f)

    gen = GenerateTrigger(15, "start", cont=False)
    trigger = gen.trigger()
    poisoned = trigger + signal
    f = PATH + f"poisoned_fft_nocont_15.png"
    plot_fft(poisoned, sample_rate, save=SAVE, f=f)
    f = PATH + f"poisoned_nocont_15.png"
    plot_waveform(poisoned, sample_rate, save=SAVE, f=f)


if __name__ == "__main__":
    try:
        test_trigger()
    except TriggerInfeasible as err:
        print(err)
