# Download the data
wget download.tensorflow.org/data/speech_commands_v0.01.tar.gz

# Extract the dataset
mkdir data
tar xzf speech_commands_v0.01.tar.gz --directory data

# Change sampling rate
cd data
cp ../change_sr.sh .
./change_sr.sh
cd -

# Preprocess data
python prepare_dataset.py mfccs data 44100 40 1103 441
