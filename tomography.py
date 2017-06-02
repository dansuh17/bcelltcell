import h5py
import numpy as np


class SampleGenerator:
    def __init__(self, filename, batch_size, use_original_sets=False):
        self.filename = filename
        self.batch_size = batch_size

        # use different split methods
        if use_original_sets:
            train_sample_ids, test_sample_ids = self.split_dataset_by_original_labels()
        else:
            train_sample_ids, test_sample_ids = self.split_dataset(ratio=0.97)

        num_train_data = len(train_sample_ids)
        self.num_batches = num_train_data // self.batch_size  # number of batches / epoch

        # trim the leftovers
        train_sample_ids = np.array(train_sample_ids[:(self.num_batches * self.batch_size)])
        self.train_sample_sets = np.split(train_sample_ids, self.num_batches)
        self.test_sample_ids = test_sample_ids
        self.batch_index = 0

        print('Train samples : {}'.format(len(train_sample_ids)))
        print('Test samples : {}'.format(len(test_sample_ids)))

    @staticmethod
    def slice_nine(sample, cut_idx: int=33):
        img_shape = (66, 66, 1)
        # 1
        z_slice = sample[cut_idx].reshape(img_shape)
        # 2
        x_slice = sample[:, :, cut_idx].reshape(img_shape)
        # 3
        y_slice = sample[:, cut_idx, :].reshape(img_shape)

        # 4
        z_y_slice = []
        for i in range(66):
            x_row = sample[i, i, :]
            z_y_slice.append(x_row)
        z_y_slice = np.array(z_y_slice).reshape(img_shape)

        # 5
        x_y_slice = []
        for i in range(66):
            z_row = sample[:, i, i]
            x_y_slice.append(z_row)
        x_y_slice = np.array(x_y_slice).reshape(img_shape)

        # 6
        x_z_slice = []
        for i in range(66):
            y_row = sample[i, :, i]
            x_z_slice.append(y_row)
        x_z_slice = np.array(x_z_slice).reshape(img_shape)

        # 7
        rev_z_y_slice = []
        for i in range(66):
            x_row = sample[i, (65 - i), :]
            rev_z_y_slice.append(x_row)
        rev_z_y_slice = np.array(rev_z_y_slice).reshape(img_shape)

        # 8
        rev_x_y_slice = []
        for i in range(66):
            z_row = sample[:, i, (65 - i)]
            rev_x_y_slice.append(z_row)
        rev_x_y_slice = np.array(rev_x_y_slice).reshape(img_shape)

        # 9
        rev_x_z_slice = []
        for i in range(66):
            y_row = sample[(65 - i), :, i]
            rev_x_z_slice.append(y_row)
        rev_x_z_slice = np.array(rev_x_z_slice).reshape(img_shape)

        nine_channel = np.concatenate([x_slice, y_slice, z_slice,
                                       z_y_slice, x_y_slice, x_z_slice,
                                       rev_z_y_slice, rev_x_y_slice,
                                       rev_x_z_slice], axis=2)
        return nine_channel

    def batch_and_label_slice(self, id_list):
        with h5py.File(self.filename, 'r') as hf:
            batch_data = []
            label_data = []

            for file_id in id_list:
                sample = hf[file_id]['data']
                # create slices with nine channels
                nine_ch_sample = self.slice_nine(sample)
                batch_data.append(nine_ch_sample)

                # add label
                label = sample.attrs['label']
                label_data.append(self.make_label(label_str=label, sparse=True))
        return batch_data, label_data

    @staticmethod
    def make_label(label_str, sparse=True):
        if sparse:
            if label_str == 'B':
                label_idx = 0
            elif label_str == 'CD4':
                label_idx = 1
            elif label_str == 'CD8':
                label_idx = 2
            else:
                raise ValueError
            return label_idx
        else:
            label_flags = [0, 0, 0]
            if label_str == 'B':
                label_flags[0] = 1
            elif label_str == 'CD4':
                label_flags[1] = 1
            elif label_str == 'CD8':
                label_flags[2] = 1
            else:
                raise ValueError
            return label_flags

    def reset_index(self):
        self.batch_index = 0

    def split_dataset_by_original_labels(self):
        train_data_idx = []
        test_data_idx = []
        with h5py.File(self.filename, 'r') as hf:
            for data_num in hf:
                original_label = hf[data_num]['data'].attrs['is_train']
                if original_label == 'train':
                    train_data_idx.append(data_num)
                elif original_label == 'test':
                    test_data_idx.append(data_num)
                else:
                    raise ValueError
        return train_data_idx, test_data_idx

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
                # 0, 1, or 2
                label_idx = self.make_label(label_str=label, sparse=True)
                label_data.append(label_idx)
        return batch_data, label_data

    def generate_samples(self):
        set_ = self.train_sample_sets[self.batch_index]
        self.batch_index += 1
        return self.batch_and_label(set_)

    def test_samples(self):
        return self.batch_and_label(self.test_sample_ids)

    def generate_sample_slices(self):
        set_ = self.train_sample_sets[self.batch_index]
        self.batch_index += 1
        return self.batch_and_label_slice(set_)

    def test_sample_slices(self, random_sample=20):
        if random_sample is not None:
            sampled_ids = np.random.choice(self.test_sample_ids, random_sample)
        else:
            sampled_ids = self.test_sample_ids
        return self.batch_and_label_slice(sampled_ids)


if __name__ == '__main__':
    sg = SampleGenerator('augmented_dataset.h5', batch_size=20)
    batch_data, batch_label = sg.generate_sample_slices()
    print(batch_data[0].shape)

