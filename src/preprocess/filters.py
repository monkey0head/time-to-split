from .utils import dataset_stats

"""
Filter interactions.
"""

import pandas as pd


def min_count_filter(data, min_count, col_name, verbose=False):
    """Filter by occurrence threshold.

    :param data: interactions log
    :param min_count: minimal number of interactions required
    :param col_name: column name, e.g. user or item id
    """
    counts = data[col_name].value_counts()
    data = data[data[col_name].isin(counts[counts >= min_count].index)]

    if verbose:
        print(dataset_stats(data, extended=True))
    return data


def drop_consecutive_repeats(
    data: pd.DataFrame, user_id="user_id", item_id="item_id", timestamp="timestamp"
):
    """Remove repeated items like i-i-j -> i-j. Keep the first consecutive interaction.

    :param data: interactions log
    :param user_id: user col name, defaults to 'user_id'
    :param item_id: item col name, defaults to 'item_id'
    :param timestamp: timestamp col name, defaults to 'timestamp'
    """

    data_sorted = data.sort_values([user_id, timestamp], kind="stable")
    data_sorted["shifted"] = data_sorted.groupby(user_id)[item_id].shift(periods=1)
    return (
        data_sorted[data_sorted[item_id] != data_sorted["shifted"]]
        .drop("shifted", axis=1)
        .reset_index(drop=True)
    )


def core_filter(
    data,
    item_min_count=5,
    seq_min_len=5,
    drop_conseq_repeats=False,
    user_id="user_id",
    item_id="item_id",
    timestamp="timestamp",
):
    """N-core filter

    :param data: _description_
    :param item_min_count: _description_, defaults to 5
    :param seq_min_len: _description_, defaults to 5
    :param drop_conseq_repeats: if remove consecutive repeated items, defaults to False
    """
    step = 1

    data = data.copy()
    if drop_conseq_repeats:
        data = drop_consecutive_repeats(data, user_id, item_id, timestamp)
        print("After consecutive repeats filtering")
        print(dataset_stats(data, extended=True))

    while len(data) > 0 and (
        data[user_id].value_counts().min() < seq_min_len
        or data[item_id].value_counts().min() < item_min_count
    ):
        data = min_count_filter(data, min_count=seq_min_len, col_name=user_id)
        data = min_count_filter(data, min_count=item_min_count, col_name=item_id)

        if drop_conseq_repeats:
            data = drop_consecutive_repeats(
                data, user_id=user_id, item_id=item_id, timestamp=timestamp
            )

        print(f"After n-core filtering on step {step}")
        print(dataset_stats(data, extended=True))
        step += 1
    return data