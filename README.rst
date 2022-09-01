Backdoor Attack with Ultrasonic Trigger
=======================================
In this project we implement a backdoor attack through an ultrasonic trigger.
This trigger is a wav file generated with SoX. To install this tool in a Debian
based Linux distro run::

  sudo apt install sox

To generate the trigger run in the commandline::

  sox -V -r 44100 -n -b 16 -c 1 trigger.wav synth 1 sin 21k vol -10dB

Using this file we implemented a trigger class that can create different
triggers based on its initialization parameters like the trigger's position,
type and duration (see trigger.py).

Procedure
---------
Use the startup.sh script downloads and extracts the dataset to the data
folder, changes the sampling rate of the audio files and runs the preprocessing
step.  In case we need only the ten classes of the dataset we should delete
everything in the data folder except the directories (“yes”, “no”, “up”,
“down”, “left”, “right”, “on”, “off”, “stop”, “go”).

The preprocessing step creates a json file that will be used by our main script
(the DATA_PATH constant has the json's name hardcoded). The script can be run
as follows::

  python run.py <experiments filename>

where <experiments file> is a csv that contains various stats about the
experiments run. It contains the architecture name, the trigger's hyperparams
(trigger type, size and position and number of poisoned samples), the attack
success rate, the accuracy for the original task, and the number of training
epochs of our model.

Android Application
-------------------
In ./app we forked tensorflow/examples (publicly available) and implemented our
android application. Our application's directory is
app/lite/examples/speech_commands/android/app/. The app is tested in a Huawei
P9 Lite mobile running Android 7 and installed in the phone with android-studio.

About
-----
This is the corresponding code for the paper "Can You Hear It? Backdoor Attacks
via Ultrasonic Triggers" that is about to be presented in WiseML '22 workshop.

To cite our work please use (this is going to be updated when the WiseML
bibtex entry is available)::

  @inproceedings{10.1145/3522783.3529523,
    author = {Koffas, Stefanos and Xu, Jing and Conti, Mauro and Picek, Stjepan},
    title = {Can You Hear It? Backdoor Attacks via Ultrasonic Triggers},
    year = {2022},
    isbn = {9781450392778},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3522783.3529523},
    doi = {10.1145/3522783.3529523},
    abstract = {This work explores backdoor attacks for automatic speech
    recognition systems where we inject inaudible triggers. By doing so, we make
    the backdoor attack challenging to detect for legitimate users and,
    consequently, potentially more dangerous. We conduct experiments on two
    versions of a speech dataset and three neural networks and explore the
    performance of our attack concerning the duration, position, and type of the
    trigger.Our results indicate that less than 1% of poisoned data is sufficient
    to deploy a backdoor attack and reach a 100% attack success rate. We observed
    that short, non-continuous triggers result in highly successful attacks. Still,
    since our trigger is inaudible, it can be as long as possible without raising
    any suspicions making the attack more effective. Finally, we conduct our attack
    on actual hardware and saw that an adversary could manipulate inference in an
    Android application by playing the inaudible trigger over the air.},
    booktitle = {Proceedings of the 2022 ACM Workshop on Wireless Security and
    Machine Learning},
    pages = {57–62},
    numpages = {6},
    keywords = {neural networks, backdoor attacks, inaudible trigger},
    location = {San Antonio, TX, USA},
    series = {WiseML '22}
  }
