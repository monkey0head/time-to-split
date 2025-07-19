"""
Preprocessing utils.
"""


def calculate_sequence_stats(lengths, prefix=''):
    """
    prefix: prefix for statistic names (e.g., 'input_' or 'gt_')
    """
    stats = {
        f'{prefix}mean': lengths.mean(),
        f'{prefix}std': lengths.std(),
        f'{prefix}min': lengths.min(),
        f'{prefix}max': lengths.max(),
        f'{prefix}median': lengths.median()
    }
    return stats


def get_time_period_days(max_timestamp, min_timestamp):
    return (max_timestamp - min_timestamp) / (60 * 60 * 24)

def dataset_stats(
    data, extended=False, user_id="user_id", item_id="item_id", timestamp="timestamp"
):
    """_summary_

    :param data: pandas dataframe
    :param extended: if calc #items per user and vise versa, defaults to False
    :param user_id: user col name, defaults to 'user_id'
    :param item_id: item col name, defaults to 'item_id'
    :param timestamp: timestamp col name, defaults to 'timestamp'
    :return: statistics dict
    """
    n_users = data[user_id].nunique()
    n_items = data[item_id].nunique()
    n_interactions = len(data)
    seq_lengths = data.groupby(user_id).size()

    stats = {
        "n_users": n_users,
        "n_items": n_items,
        "n_interactions": n_interactions,
        "density": n_interactions / (n_users * n_items),
        "avg_seq_length": seq_lengths.mean(),
    }

    if extended:
        stats.update(calculate_sequence_stats(seq_lengths, prefix='seq_len_'))
        
        item_counts = data[item_id].value_counts()
        stats.update(calculate_sequence_stats(item_counts, prefix='item_occurrence_'))
        
        user_counts = data[user_id].value_counts()
        stats.update(calculate_sequence_stats(user_counts, prefix='user_activity_'))
        
        # Temporal statistics
        stats.update({"max_timestamp": data[timestamp].max(), 
                      "min_timestamp": data[timestamp].min()})

        stats["timestamp_range_in_days"] = get_time_period_days(stats["max_timestamp"], 
                                                                stats["min_timestamp"]) 
        
        duration_user = data.groupby(user_id)[timestamp].agg(min_ts='min', max_ts='max')
        duration_user['duration'] = get_time_period_days(duration_user['max_ts'], 
                                                         duration_user['min_ts'])
        
        stats.update({"mean_user_duration": duration_user['duration'].mean(), 
                      "median_user_duration": duration_user['duration'].median()})
    
    return stats


def rename_cols(data, user_id="user_id", item_id="item_id", timestamp="timestamp"):
    "Rename columns of dataframe"

    data = data.rename(
        columns={user_id: "user_id", item_id: "item_id", timestamp: "timestamp"}
    )

    return data