"""Make statistics of train, validation and test."""

import os
import pickle

import hydra
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
from clearml import Task
from src.preprocess.utils import dataset_stats, calculate_sequence_stats, get_time_period_days, rename_cols
from src.prepr import last_item_split, global_temporal_split

import warnings
warnings.filterwarnings("ignore")

def init_clearml_task(project_name, task_name, config):
    """Initialize ClearML task and logger."""
    task = Task.init(
        project_name=project_name, task_name=task_name, reuse_last_task_id=False
    )
    task.connect(OmegaConf.to_container(config, resolve=True))
    return task, task.get_logger()


def log_stats_to_clearml(stats, config, stage):
    """Log statistics to ClearML as individual values."""
    project_name = os.path.join(config.clearml_project_folder, "statistics", stage)
    task_name = f"{config.dataset.name}_{config.clearml_task_name}_{stage}"
    task, logger = init_clearml_task(project_name, task_name, config)
    for key, value in stats.items():
        logger.report_single_value(f"{stage}_stats/{key}", value)
    task.close()

def build_splitted_data_path(config, prefix=None):
    prefix = os.environ["SEQ_SPLITS_DATA_PATH"] if prefix is None else prefix
    validation_type = config.split_params.validation_type or ""
    q = (
        "q0" + str(config.split_params.quantile)[2:]
        if config.split_type == "global_timesplit"
        else ""
    )
    return os.path.join(
            prefix,
            "splitted",
            config.split_type,
            validation_type,
            config.dataset.name,
            q,
    )

def save_stats_to_csv(stats, config, subset):
    """Save statistics to a CSV file."""
    if subset == "raw" or subset == "preprocessed":
        results_path = os.path.join(
        os.path.dirname(__file__), '../data', 'statistics', subset)
        file_name = f"{config.dataset.name}.csv"
    else:
        results_path = build_splitted_data_path(config, prefix=os.path.join(
        os.path.dirname(__file__), '../data', 'statistics'))
        file_name = f"{subset}_stats.csv"
 
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    stats_df = pd.DataFrame(stats, index=[0])  # Convert stats dictionary to DataFrame
    stats_df.to_csv(os.path.join(results_path, file_name), index=False)


def get_time_idx_reversed(df, user_col="user_id", timestamp_col="timestamp"):
    df = df.sort_values([user_col, timestamp_col], kind="stable")
    df["time_idx_reversed"] = df.groupby(user_col).cumcount(ascending=False)
    return df


def get_two_last(df, user_col="user_id", timestamp_col="timestamp"):
    df = get_time_idx_reversed(df, user_col, timestamp_col)
    return df[df["time_idx_reversed"] < 2]


def get_deltas(df):
    df.sort_values(by=["timestamp"], kind="stable", inplace=True)
    df["next_ts"] = df.groupby("user_id")["timestamp"].shift(-1)
    df = df.dropna()
    df["next_ts"] = df["next_ts"].astype(int)
    df["delta"] = (df["next_ts"] - df["timestamp"]).astype(int)
    return df


def count_delta_stats(data, prefix):
    deltas = get_deltas(data)
    return {
        f"{prefix}_mean_delta": deltas["delta"].mean(),
        f"{prefix}_median_delta": deltas["delta"].median(),
    }



def count_gt(data, stage, config):
    """Calculate statistics for validation/test splits considering ground truth length."""
    stats = {}
    splitted_data_path = build_splitted_data_path(config)
    
    if (stage == "test" and config.split_type == "global_timesplit") or \
        (stage == "validation" and config.split_params.validation_type == "val_by_time"):
        
        time_threshold_name = "time_threshold.pkl"
        if stage == "validation":
            time_threshold_name = "val_time_threshold.pkl"
    
        time_threshold = pickle.load(open(os.path.join(splitted_data_path, time_threshold_name), "rb"))
        df_input, df_holdout = global_temporal_split(
            data, time_threshold, remove_cold=False
        )
        stats.update(calculate_sequence_stats(df_input.groupby("user_id").size(), prefix="input_seq_len_"))
        stats.update(calculate_sequence_stats(df_holdout.groupby("user_id").size(), prefix="holdout_seq_len_"))
        
        stats['range_gt_tmstmp_delta_in_days'] = get_time_period_days(
            df_holdout["timestamp"].max(), time_threshold)
        
        df_input = get_time_idx_reversed(df_input)
        df_holdout = get_time_idx_reversed(df_holdout)

        last_input_item = (
            df_input[df_input["time_idx_reversed"] == 0]
            .drop(columns=["time_idx_reversed"])
            .reset_index(drop=True)
        )

        # diffrent gts, gt item plus all previous
        holdout_data = {}
        holdout_data["first"] = df_holdout[
            df_holdout["time_idx_reversed"]
            == df_holdout.groupby("user_id")["time_idx_reversed"].transform(max)
        ]
        holdout_data["successive"] = df_holdout

        np.random.seed(config.random_state)
        random_gt_position = (
            df_holdout.groupby("user_id")["time_idx_reversed"]
            .max()
            .apply(lambda x: np.random.randint(0, x + 1))
        )
        holdout_data["random"] = df_holdout[
            df_holdout["time_idx_reversed"]
            >= df_holdout["user_id"].map(random_gt_position)
        ]

        for current_gt_name, current_df in holdout_data.items():
            concatenated = pd.concat(
                [last_input_item, current_df[["user_id", "item_id", "timestamp"]]],
                ignore_index=True,
            )
            if current_gt_name == "successive":
                stats.update(
                    count_delta_stats(get_two_last(concatenated), prefix="last")
                )
                stats.update(count_delta_stats(concatenated, prefix=current_gt_name))
                stats["last_gt_mean_position"] = (
                    current_df.groupby("user_id").size().mean()
                )
            else:
                stats.update(
                    count_delta_stats(
                        get_two_last(concatenated), prefix=current_gt_name
                    )
                )
                stats[f"{current_gt_name}_gt_mean_position"] = (
                    current_df.groupby("user_id").size().mean()
                )


    elif config.split_type == "leave-one-out" or \
        (stage == "validation" and config.split_params.validation_type != "val_by_time"):
        input_seqs, gt_items = last_item_split(data)
        user_lengths = input_seqs.groupby("user_id").size()
        stats.update(calculate_sequence_stats(user_lengths, prefix="input_seq_len_"))
        stats['range_gt_tmstmp_delta_in_days'] = get_time_period_days(
            gt_items["timestamp"].max(), gt_items["timestamp"].min())
        
        df_input = get_time_idx_reversed(input_seqs)

        last_input_item = (
            df_input[df_input["time_idx_reversed"] == 0]
            .drop(columns=["time_idx_reversed"])
            .reset_index(drop=True)
        )

        concatenated = pd.concat(
            [last_input_item, gt_items[["user_id", "item_id", "timestamp"]]],
            ignore_index=True,
        )
        stats.update(
            count_delta_stats(get_two_last(concatenated), prefix="last")
        )
        
    else:
        raise ValueError("Wrong type of splitter.")
    return stats


def process_dataset_stats(data, config, stage):
    """Process and log statistics for a dataset stage (raw/preprocessed)."""

    stats = dataset_stats(data, extended=True)
    stats.update(count_delta_stats(data, prefix=stage))
    
    if stage == "validation" or stage == "test":
        stats.update(count_gt(data, stage, config))

    to_print = []
    for key in [config.dataset.name, config.split_type, 
                config.split_params.validation_type, config.split_params.quantile, stage, "statistics:"]:

        if key is not None:
            to_print.append(str(key))
    to_print = " ".join(to_print)    
    print(to_print)
    print(stats)

    if config.clearml_project_folder is not None:
        log_stats_to_clearml(stats, config, stage)

    save_stats_to_csv(stats, config, stage)

    return stats


@hydra.main(version_base=None, config_path="configs", config_name="statistics")
def main(config):
    print(OmegaConf.to_yaml(config, resolve=True))

    data_path = os.environ["SEQ_SPLITS_DATA_PATH"]
    if config.split_type == "raw" or config.split_type == "preprocessed":
        data = pd.read_csv(
            os.path.join(data_path, config.split_type, config.dataset.name + ".csv")
        )
        if config.split_type == "raw":
            columns = [
                new for _, new in config.dataset.column_name.items() if new is not None
            ]
            data = rename_cols(data[columns].copy(), *columns)
        process_dataset_stats(data, config, config.split_type)

    else:
        splitted_data_path = build_splitted_data_path(config)
        split_train = pd.read_csv(os.path.join(splitted_data_path, "train.csv"))
        split_validation = pd.read_csv(
            os.path.join(splitted_data_path, "validation.csv")
        )
        split_test = pd.read_csv(os.path.join(splitted_data_path, "test.csv"))

        for split_data, stage in [
            (split_train, "train"),
            (split_validation, "validation"),
            (split_test, "test"),
        ]:
            process_dataset_stats(split_data, config, stage)


if __name__ == "__main__":
    main()
