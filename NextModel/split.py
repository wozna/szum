import os, shutil
from random import shuffle


def create_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def main(original_dataset_dir, base_dir):
    create_dir(base_dir)
    train_dir = os.path.join(base_dir, 'training')
    create_dir(train_dir)
    validation_dir = os.path.join(base_dir, 'validation')
    create_dir(validation_dir)
    test_dir = os.path.join(base_dir, 'evaluation')
    create_dir(test_dir)

    list_origin_dirs = []

    list_train_dirs = []
    list_validation_dirs = []
    list_test_dirs = []

    # create dirs of elements
    for dir in os.listdir(original_dataset_dir):
        origin_dir_element = os.path.join(original_dataset_dir, dir)
        list_origin_dirs.append(origin_dir_element)

        train_dir_element = os.path.join(train_dir, dir)
        list_train_dirs.append(train_dir_element)
        create_dir(train_dir_element)

        validation_dir_element = os.path.join(validation_dir, dir)
        list_validation_dirs.append(validation_dir_element)
        create_dir(validation_dir_element)

        test_dir_element = os.path.join(test_dir, dir)
        list_test_dirs.append(test_dir_element)
        create_dir(test_dir_element)

    train_words = ['yes', 'no', 'up', 'down',
                   'left', 'right', 'on', 'off', 'stop', 'go']

    # copy elements from origin dirs to train, validation and test dirs
    for i, path in enumerate(list_origin_dirs):
        files = os.listdir(path)
        shuffle(files)
        print(str(i) + " of " + str(len(list_origin_dirs)))
        total_files = len(files)
        print(total_files)
        file_path = ""
        dir_name = os.path.basename(path)
        print(dir_name)
        iter = 1
        if dir_name in train_words:
            iter = 10

        for j, file in enumerate(files):
            origin_file_path = os.path.join(path, file)
            if j < 119 * iter:
                file_path = os.path.join(list_train_dirs[i], file)
            elif j < 153 * iter:
                file_path = os.path.join(list_validation_dirs[i], file)
            elif j < 170 * iter:
                file_path = os.path.join(list_test_dirs[i], file)
            shutil.copy(origin_file_path, file_path)
    print("finish")


if __name__ == "__main__":
    print("Skrypt do podziału zbioru danych na zbiory do uczenia, walidacji i testów")
    original_dataset_dir = str(input("Wpisz ścieżkę do zbioru który ma być podzielony (original_dataset_dir): "))
    base_dir = str(
        input("Wpisz ścieżkę do folderu, w którym ma być zapisany podzielony zbiór 'original_dataset_dir': "))
    main(original_dataset_dir, base_dir)
