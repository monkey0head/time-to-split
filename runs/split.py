"""Make split to train, validation and test."""

import os

import hydra
import pandas as pd
from omegaconf import OmegaConf
import pickle

from src.preprocess.utils import dataset_stats
from src.splits import LeaveOneOutSplitter, GlobalTimeSplitter


@hydra.main(version_base=None, config_path="configs", config_name="split")
def main(config):

    print(OmegaConf.to_yaml(config, resolve=True))

    data_path = os.environ["SEQ_SPLITS_DATA_PATH"]
    data = pd.read_csv(os.path.join(data_path, 'preprocessed', config.dataset.name + '.csv'))

    if config.split_type == 'leave-one-out':
        splitter = LeaveOneOutSplitter()
        dir_name = os.path.join(data_path, "splitted", config.split_type, config.dataset.name)
        train, validation, test = splitter.split(data)
    elif config.split_type == 'global_timesplit':
        splitter = GlobalTimeSplitter(**config.split_params)
        if config.split_params.quantile is not None:
            q = 'q0'+str(config.split_params.quantile)[2:]
        else:
            raise ValueError("'global_timesplit' split must be run with parameter 'quantile'")
        dir_name = os.path.join(data_path, "splitted", config.split_type,
                                'val_' + config.split_params.validation_type,
                                config.dataset.name, q)
        train, validation, test, time_threshold, val_time_threshold = splitter.split(data)
    else:
        raise ValueError('Wrong type of splitter.')

    print("train statistics")
    print(dataset_stats(train, extended=True))
    print("validation statistics")
    print(dataset_stats(validation, extended=True))
    print("test statistics")
    print(dataset_stats(test, extended=True))

    if config.save_results:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        train.to_csv(os.path.join(dir_name, 'train.csv'), index=False)
        validation.to_csv(os.path.join(dir_name, 'validation.csv'), index=False)
        test.to_csv(os.path.join(dir_name, 'test.csv'), index=False)
        if config.split_type == 'global_timesplit':
            with open(os.path.join(dir_name, 'time_threshold.pkl'), 'wb') as f:
                pickle.dump(time_threshold, f)
            print('Global test timepoint:', time_threshold)
            if config.split_params.validation_type == 'by_time':
                with open(os.path.join(dir_name, 'val_time_threshold.pkl'), 'wb') as f:
                    pickle.dump(val_time_threshold, f)
                print('Global validation timepoint:', val_time_threshold)

if __name__ == "__main__":

    main()