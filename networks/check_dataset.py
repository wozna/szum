import os
import sys
import shutil

directory = os.path.abspath(os.getcwd()) + "/commands"

print(directory)
train_words = ['yes', 'no', 'up', 'down', '_background_noise_',
               'left', 'right', 'on', 'off', 'stop', 'go']

for subdir, dirs, files in os.walk(directory):
    for dir in dirs:
        print(dir)
        if dir not in train_words:
            try:
                print("removing " + dir)
                shutil.rmtree(directory+'/'+dir)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))
