import pathlib
import random
import time


class Timer:
    """
    Timer context manager for control execution time.
    """
    def __init__(self, name: str):
        self.name = name
        self.start = None
        self.end = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        print('- Timer-{name} has expired in {duration}s.'.format(
            name=self.name,
            duration=self.end-self.start
        ))

    def msg(self, text: str):
        print(text)
        return self


class DataLoader(object):
    """
    Handles all aspects of the data. Stores the dataset_params.
    """
    def __init__(self, params: dict):
        """
        Loads dataset_params.

        :param params: dictionary with attributes
        """
        self.data_params = params
        self.data_dir = None

    def set_dir(self, data_dir: str):
        """
        Set up directory with data.

        :param data_dir:
        :return: self
        """
        self.data_dir = pathlib.Path(data_dir)
        return self

    def load_data(self, types: list, data_dir: str):
        """
        Loads the data for each type in types from data_dir.

        :param types:
        :param data_dir:
        :return: dict that contains the data with labels for each type in types
        """
        dir_ = pathlib.Path(data_dir)
        data = {}

        for split in ['train', 'dev', 'test']:
            if split in types:
                text = (dir_/"sentences.txt") \
                    .read_text(encoding='utf-8') \
                    .split('\n')
                labels = (dir_/"labels.txt") \
                    .read_text(encoding='utf-8') \
                    .split('\n')

                assert len(text) == len(labels)

                data[split] = {
                    'text': text,
                    'labels': labels,
                    'size': len(text)
                }

        return data

    def data_iterator(self, data: dict, params: dict, shuffle: bool = False):
        """
        Returns a generator that yields batches data with labels.
        Batch size is params.batch_size. Expires after one pass over the data.

        :param data: contains keys 'text', 'labels' and 'size'
        :param params: dictionary of attributes
        :param shuffle: whether the data should be shuffled
        """
        order = list(range(data['size']))
        batch_size = params.get('batch_size', 100)

        if shuffle:
            random.seed(493)
            random.shuffle(order)

        for i in range((data['size'] + 1)//batch_size):
            # fetch text and labels
            batch_text = [data['text'][idx] for idx in order[i*batch_size:(i+1)*batch_size]]
            batch_labels = [data['labels'][idx] for idx in order[i*batch_size:(i+1)*batch_size]]

            yield batch_text, batch_labels

