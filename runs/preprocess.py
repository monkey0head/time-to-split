"""
Preprocessing pipeline
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
from omegaconf import OmegaConf, open_dict

import pandas as pd

from src.preprocess.filters import core_filter
from src.preprocess.indexes import encode
from src.preprocess.utils import dataset_stats, rename_cols


def preprocess(
    data,
    item_min_count=5,
    seq_min_len=5,
    core=True,
    encoding=True,
    drop_conseq_repeats=False,
    filter_by_relevance=False,
    users_sample=None,
    user_id="user_id",
    item_id="item_id",
    timestamp="timestamp",
    relevance="relevance",
    path_to_save=None,
):
    """
    - columns renaming
    - N-core or N-filter for items and sequences along with iterative
    - removal of consecutive interactions with the same item
    - label encoding of users and items, item labels starts with 1 to leave 0 as a padding value
    """
    # filter columns TO DO: make optional
    columns = [
        key for key in [user_id, item_id, timestamp, relevance] if key is not None
    ]

    data = data[columns].copy()

    data = rename_cols(data, user_id, item_id, timestamp)

    print("Raw data")
    print(dataset_stats(data, extended=True))

    if filter_by_relevance:
        raise NotImplementedError("No filter_by_relevance implemented")

    if users_sample is not None:
        raise NotImplementedError("No user sampling implemented")

    if core:
        data = core_filter(
            data=data,
            item_min_count=item_min_count,
            seq_min_len=seq_min_len,
            drop_conseq_repeats=drop_conseq_repeats,
            user_id="user_id",
            item_id="item_id",
            timestamp="timestamp",
        )
        print("After N-core")
        print(dataset_stats(data, extended=True))

    else:
        raise NotImplementedError("N-core filtering is only one available")

    if encoding:
        data = encode(data=data, col_name="user_id", shift=0)
        data = encode(data=data, col_name="item_id", shift=1)

    if path_to_save is not None:
        data.to_csv(path_to_save, index=False)

    return data


@hydra.main(config_path="configs", config_name="preprocess")
def main(config):
    data_format = "csv"
    print(OmegaConf.to_yaml(config, resolve=True))
    data_path = os.environ["SEQ_SPLITS_DATA_PATH"]
    data = pd.read_csv(
        os.path.join(data_path, "raw", f"{config.dataset.name}.{data_format}")
    )

    with open_dict(config):
        save_to_disk = config.prep_params.pop("save_to_disk")

    preprocess(
        data=data,
        **config.prep_params,
        **config.dataset.column_name,
        path_to_save=os.path.join(
            data_path, "preprocessed", f"{config.dataset.name}.{data_format}"
        )
        if save_to_disk
        else None,
    )


if __name__ == "__main__":
    main()