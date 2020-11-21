#noises 

wget https://zenodo.org/record/400515/files/TUT-acoustic-scenes-2017-development.audio.3.zip?download=1
wget https://zenodo.org/record/400515/files/TUT-acoustic-scenes-2017-development.audio.2.zip?download=1
wget https://zenodo.org/record/400515/files/TUT-acoustic-scenes-2017-development.audio.1.zip?download=1

unzip TUT-acoustic-scenes-2017-development.audio.1.zip?download=1
unzip TUT-acoustic-scenes-2017-development.audio.2.zip?download=1
unzip TUT-acoustic-scenes-2017-development.audio.3.zip?download=1
mkdir noise
mv TUT-acoustic-scenes-2017-development noise


#commands

wget http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz
mkdir commands
tar -xzf speech_commands_v0.01.tar.gz -C commands

#remove unnecessary commands
python3 check_dataset.py