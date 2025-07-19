"""
Torch datasets and collate function.
"""

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class LMDataset(Dataset):

    def __init__(self, df, max_length=128, num_negatives=None, full_negative_sampling=False,
                 user_col='user_id', item_col='item_id', time_col='timestamp'):

        self.max_length = max_length
        self.num_negatives = num_negatives
        self.full_negative_sampling = full_negative_sampling
        self.user_col = user_col
        self.item_col = item_col
        self.time_col = time_col

        self.data = df.sort_values(time_col, kind="stable").groupby(user_col)[item_col].agg(list).to_dict()
        self.user_ids = list(self.data.keys())

        if num_negatives:
            self.all_items = df[item_col].unique()

    def __len__(self):

        return len(self.data)

    def sample_negatives(self, item_sequence):

        negatives = self.all_items[~np.isin(self.all_items, item_sequence)]
        if self.full_negative_sampling:
            negatives = np.random.choice(
                negatives, size=self.num_negatives * (len(item_sequence) - 1), replace=True)
            negatives = negatives.reshape(len(item_sequence) - 1, self.num_negatives)
        else:
            # replace=True for speed, with replace=False sampling can be very slow
            negatives = np.random.choice(negatives, size=self.num_negatives, replace=True)

        return negatives


class CausalLMDataset(LMDataset):

    def __init__(self, df, max_length=128,
                 shift_labels=True, num_negatives=None,
                 full_negative_sampling=False,
                 user_col='user_id', item_col='item_id',
                 time_col='timestamp'):

        super().__init__(df, max_length, num_negatives, full_negative_sampling,
                         user_col, item_col, time_col)

        self.shift_labels = shift_labels

    def __getitem__(self, idx):

        item_sequence = self.data[self.user_ids[idx]]

        if len(item_sequence) > self.max_length + 1:
            item_sequence = item_sequence[-self.max_length - 1:]

        input_ids = np.array(item_sequence[:-1])
        if self.shift_labels:
            labels = np.array(item_sequence[1:])
        else:
            labels = input_ids

        if self.num_negatives:
            negatives = self.sample_negatives(item_sequence)
            return {'input_ids': input_ids, 'labels': labels, 'negatives': negatives}

        return {'input_ids': input_ids, 'labels': labels}


class CausalLMPredictionDataset(LMDataset):

    def __init__(self, df, max_length=128, validation_mode=False,
                 user_col='user_id', item_col='item_id',
                 time_col='timestamp'):

        super().__init__(df, max_length=max_length, num_negatives=None,
                         user_col=user_col, item_col=item_col, time_col=time_col)

        self.validation_mode = validation_mode

    def __getitem__(self, idx):

        user_id = self.user_ids[idx]
        item_sequence = self.data[user_id]

        if self.validation_mode:
            target = item_sequence[-1]
            input_ids = item_sequence[-self.max_length-1:-1]
            item_sequence = item_sequence[:-1]

            return {'input_ids': input_ids, 'user_id': user_id,
                    'seen_ids': item_sequence, 'target': target}
        else:
            input_ids = item_sequence[-self.max_length:]

            return {'input_ids': input_ids, 'user_id': user_id,
                    'seen_ids': item_sequence}
    

class SuccessivePredictionDataset(LMDataset):

    def __init__(self, df, global_timepoint, max_length=128,
                 user_col='user_id', item_col='item_id',
                 time_col='timestamp',
                 masking_value=None, padding_value=0):

        super().__init__(df, max_length=max_length, num_negatives=None,
                         user_col=user_col, item_col=item_col, time_col=time_col)
        
        self.masking_value = masking_value
        self.maxlen = max_length
        self.padding_value = padding_value

        history_df = df[df[time_col] <= global_timepoint]
        test_df = df[df[time_col] > global_timepoint]

        self.data_train = history_df.sort_values(time_col, kind="stable").groupby(user_col)[item_col].agg(list).to_dict()
        self.data_test = test_df.sort_values(time_col, kind="stable").groupby(user_col)[item_col].agg(list).to_dict()

        test_user_ids = list(self.data_test.keys())

        self.test_user_ids = [
            u for u in test_user_ids 
            if (u in self.data_train) or (len(self.data_test[u]) > 1)
        ]
        
        input_ids = []
        target_ids = []
        user_ids = []
        seen_ids = []
        for idx in range(len(self.test_user_ids)):
            user_id = self.test_user_ids[idx]
            test_ids = self.data_test[user_id]

            try:
                seen_ids_user = self.data_train[user_id]
            except KeyError:                        # handle users with no history - advance by 1 item
                seen_ids_user = test_ids[:1]
                test_ids = test_ids[1:]

            input_ids_user = self.get_successive_user_seqs(test_ids[:-1], seen_ids_user)
            if self.masking_value is not None:
                input_ids_user = self.get_mask_seqs(input_ids_user, masking_value=self.masking_value)

            seen_ids_user = [seen_ids_user + test_ids[:k] for k in range(len(test_ids))]

            input_ids.extend(input_ids_user)
            target_ids.extend(test_ids)
            user_ids.extend([user_id] * len(test_ids))
            seen_ids.extend(seen_ids_user) 

        self.input_ids = torch.stack(input_ids).tolist()
        self.target_ids = target_ids
        self.user_ids = user_ids
        self.seen_ids = seen_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):

        input_ids = self.input_ids[idx]
        target_ids = self.target_ids[idx]
        user_id = self.user_ids[idx]
        seen_ids = self.seen_ids[idx]

        return {'input_ids': input_ids, 'user_id': user_id,
                'seen_ids': seen_ids, 'target_ids': target_ids}

    def get_successive_user_seqs(self, input_ids, seen_ids): # only for padding_idx = 0

        n_seen = len(seen_ids)
        n_targets = len(input_ids)
        n_rows = n_targets + 1

        seq = np.concatenate([seen_ids, input_ids])
        with torch.no_grad():
            pad_seq = torch.as_tensor(np.concatenate([seq, np.zeros(self.maxlen, dtype=seq.dtype)]), 
                                        dtype=torch.int64)

            rows = torch.arange(n_rows).unsqueeze(1)    # shape (n_rows, 1)
            cols = torch.arange(self.maxlen).unsqueeze(0)      # shape (1, maxlen)

            valid_length = torch.clamp(n_seen + rows, max=self.maxlen)  # how many tokens should be nonzero
            offset = torch.clamp(n_seen + rows - self.maxlen, min=0)       # row-specific offset

            indices = offset + cols  # shape (n_rows, maxlen)

        return torch.where(cols < valid_length, pad_seq[indices], torch.tensor(0, dtype=torch.int64))
    
    def get_mask_seqs(self, input_ids, masking_value):

        mask_rows = (input_ids[:, -1] == 0)  # shape (n_rows,)
        if mask_rows.any():
            zero_bool = (input_ids == 0)  # shape (n_rows, maxlen)
            first_zero_idx = zero_bool.float().argmax(dim=1)  # shape (n_rows,)
            input_ids[mask_rows, first_zero_idx[mask_rows]] = masking_value

        if (~mask_rows).any():
            rolled = torch.roll(input_ids[~mask_rows], shifts=-1, dims=1)
            rolled[:, -1] = masking_value
            input_ids[~mask_rows] = rolled
        
        return input_ids


class MaskedLMDataset(LMDataset):

    def __init__(self, df, max_length=128,
                 mlm_probability=0.2, num_negatives=None,
                 full_negative_sampling=False,
                 masking_value=1, ignore_value=-100,
                 force_last_item_masking_prob=0,
                 user_col='user_id', item_col='item_id',
                 time_col='timestamp'):

        super().__init__(df, max_length, num_negatives, full_negative_sampling,
                         user_col, item_col, time_col)

        self.mlm_probability = mlm_probability
        self.masking_value = masking_value
        self.ignore_value = ignore_value
        self.force_last_item_masking_prob = force_last_item_masking_prob

    def __getitem__(self, idx):

        item_sequence = self.data[self.user_ids[idx]]

        if len(item_sequence) > self.max_length:
            item_sequence = item_sequence[-self.max_length:]

        input_ids = np.array(item_sequence)
        mask = np.random.rand(len(item_sequence)) < self.mlm_probability
        input_ids[mask] = self.masking_value
        if self.force_last_item_masking_prob > 0:
            if np.random.rand() < self.force_last_item_masking_prob:
                input_ids[-1] = self.masking_value

        labels = np.array(item_sequence)
        labels[input_ids != self.masking_value] = self.ignore_value

        if self.num_negatives:
            negatives = self.sample_negatives(item_sequence)
            return {'input_ids': input_ids, 'labels': labels, 'negatives': negatives}

        return {'input_ids': input_ids, 'labels': labels}


class MaskedLMPredictionDataset(LMDataset):

    def __init__(self, df, max_length=128, masking_value=1,
                 validation_mode=False,
                 user_col='user_id', item_col='item_id',
                 time_col='timestamp'):

        super().__init__(df, max_length=max_length, num_negatives=None,
                         user_col=user_col, item_col=item_col, time_col=time_col)

        self.masking_value = masking_value
        self.validation_mode = validation_mode

    def __getitem__(self, idx):

        user_id = self.user_ids[idx]
        item_sequence = self.data[user_id]

        if self.validation_mode:
            target = item_sequence[-1]
            input_ids = item_sequence[-self.max_length:-1]
            item_sequence = item_sequence[:-1]
        else:
            input_ids = item_sequence[-self.max_length + 1:]

        input_ids += [self.masking_value]

        if self.validation_mode:
            return {'input_ids': input_ids, 'user_id': user_id,
                    'seen_ids': item_sequence, 'target': target}
        else:
            return {'input_ids': input_ids, 'user_id': user_id,
                    'seen_ids': item_sequence}
        

class PaddingCollateFn:

    def __init__(self, padding_value=0, left_padding=False,
                 labels_padding_value=-100, labels_keys=['labels']):

        self.padding_value = padding_value
        self.left_padding = left_padding
        self.labels_padding_value = labels_padding_value
        self.labels_keys = labels_keys

    def __call__(self, batch):

        collated_batch = {}

        for key in batch[0].keys():

            if np.isscalar(batch[0][key]):
                collated_batch[key] = torch.tensor([example[key] for example in batch])
                continue

            if key in self.labels_keys:
                padding_value = self.labels_padding_value
            else:
                padding_value = self.padding_value
            # left padding is required for sequence generation with huggingface models
            if self.left_padding:
                values = [torch.tensor(example[key][::-1].copy()) for example in batch]
                collated_batch[key] = pad_sequence(values, batch_first=True,
                                                   padding_value=padding_value).flip(-1)
            else:
                values = [torch.tensor(example[key]) for example in batch]
                collated_batch[key] = pad_sequence(values, batch_first=True,
                                                   padding_value=padding_value)
        if 'input_ids' in collated_batch:
            attention_mask = collated_batch['input_ids'] != self.padding_value
            collated_batch['attention_mask'] = attention_mask.to(dtype=torch.float32)

        return collated_batch
