import warnings

def last_item_split(df, user_col='user_id', timestamp_col='timestamp'):
    """Split user sequences to input data and ground truth with one last item."""

    if df[user_col].value_counts().min() == 1:
         warnings.warn('Each user must have at least two interactions.')

    df = df.sort_values([user_col, timestamp_col], kind='stable')
    df['time_idx_reversed'] = df.groupby(user_col).cumcount(ascending=False)

    inputs = df[df['time_idx_reversed'] >= 1]
    last_item = df[df['time_idx_reversed'] == 0]

    inputs = inputs.drop(columns=['time_idx_reversed']).reset_index(drop=True)
    last_item = last_item.drop(columns=['time_idx_reversed']).reset_index(drop=True)

    return inputs, last_item

def global_temporal_split(df, global_timepoint, user_col='user_id', timestamp_col='timestamp', remove_cold=True):
    if df[user_col].value_counts().min() == 1:
        warnings.warn('Each user must have at least two interactions.')

    df_gt = df[df[timestamp_col] > global_timepoint].reset_index(drop=True)
    df_input = df[df[timestamp_col] <= global_timepoint]
    # predict for test users only
    df_input = df_input[df_input[user_col].isin(df_gt[user_col].unique())].reset_index(drop=True)
    # remove cold
    if remove_cold:
        df_gt = df_gt[df_gt[user_col].isin(df_input[user_col].unique())].reset_index(drop=True)
    return df_input, df_gt

def get_ground_truth_from_successive(recs):
    return (
        recs[["user_id", "pseudo_user_id", "target"]]
        .drop_duplicates()
        .rename(columns={"target": "item_id"})
        .reset_index(drop=True)
    )
