import sys

import csv
import pathlib


def load_dataset(path_csv):
    """Loads dataset into memory from csv file

    :param path_csv: str
    :return: object - [('text..', 'spam'), ('text...', 'ham'), ...]
    """
    dataset = []
    try:
        raw_data = pathlib \
            .Path(path_csv) \
            .read_text(encoding='utf-8') \
            .split('\n')
        csv_file = csv.reader(raw_data, delimiter=',')
    except FileNotFoundError as e:
        print("Sorry, no such file or directory.")
        return dataset

    # Read data from file
    for idx, row in enumerate(csv_file):
        if idx == 0 or len(row) < 2: continue
        text, label = row
        if text:
            dataset.append((text, label))

    return dataset


def save_dataset(dataset, save_dir):
    """Writes sentences.txt and labels.txt files in save_dir from dataset

    :param dataset: object - [('text..', 'spam'), ('text...', 'ham'), ...]
    :param save_dir: str
    """

    # Create directory if not exists
    print("Saving data in {}...".format(save_dir))
    dir_ = pathlib.Path(save_dir)
    dir_.mkdir(parents=True, exist_ok=True)

    file_sentences = dir_/'sentences.txt'
    file_labels = dir_/'labels.txt'

    # Will record values into file
    for text, label in dataset:
        file_sentences.write_text("{}\n".format(text))
        file_labels.write_text("{}\n".format(label))

    print("- done.")


if __name__ == '__main__':
    dir_ = pathlib.Path('data/MMD')

    # Load dataset into memory
    print("Loading dataset into memory...")
    dataset = load_dataset(dir_ / "MMD_DS_test.csv")
    print("- done.")

    # Split the dataset into train, dev and split (dummy split with no shuffle)
    train_dataset = dataset[:int(0.7 * len(dataset))]
    dev_dataset = dataset[int(0.7 * len(dataset)): int(0.85 * len(dataset))]
    test_dataset = dataset[int(0.85 * len(dataset)):]

    # Save the datasets to files
    save_dataset(train_dataset, dir_/'train')
    save_dataset(dev_dataset, dir_/'dev')
    save_dataset(test_dataset, dir_/'test')
