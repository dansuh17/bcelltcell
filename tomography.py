import h5py
import numpy as np


class SampleGenerator:
    def __init__(self, filename, batch_size):
        self.filename = filename
        self.batch_size = batch_size

        train_sample_ids, test_sample_ids = self.split_dataset(ratio=0.96)
        num_train_data = len(train_sample_ids)
        self.num_batches = num_train_data // self.batch_size

        # trim the leftovers
        train_sample_ids = train_sample_ids[:(self.num_batches * self.batch_size)]
        self.train_sample_sets = np.split(train_sample_ids, self.num_batches)
        self.test_sample_ids = test_sample_ids
        self.batch_index = 0

        print('Train samples : {}'.format(len(train_sample_ids)))
        print('Test samples : {}'.format(len(test_sample_ids)))

    def reset_index(self):
        self.batch_index = 0

    def split_dataset(self, ratio):
        with h5py.File(self.filename, 'r') as hf:
            data_nums = [num for num in hf]
        num_samples = len(data_nums)
        split_point = int(num_samples * ratio)
        return np.split(np.random.permutation(data_nums), [split_point])

    def batch_and_label(self, id_list):
        with h5py.File(self.filename, 'r') as hf:
            batch_data = []
            label_data = []

            for samp_id in id_list:
                sample = np.array(hf[samp_id]['data'])
                sample = sample.reshape(sample.shape + (1, ))
                batch_data.append(sample)

                label = hf[samp_id]['data'].attrs['label']
                # label_flag = [0, 0, 0]
                if label == 'B':
                    label_idx = 0
                elif label == 'CD4':
                    label_idx = 1
                elif label == 'CD8':
                    label_idx = 2
                else:
                    raise ValueError
                label_data.append(label_idx)
        return batch_data, label_data

    def generate_samples(self):
        set_ = self.train_sample_sets[self.batch_index]
        self.batch_index += 1
        return self.batch_and_label(set_)

    def test_samples(self):
        return self.batch_and_label(self.test_sample_ids)

