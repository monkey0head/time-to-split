"""Data splits."""

import numpy as np


class LeaveOneOutSplitter:
    """Leave-one-out split."""

    def __init__(self, user_col="user_id", timestamp_col="timestamp"):
        self.user_col = user_col
        self.timestamp_col = timestamp_col

    def split(self, data):
        data = data.sort_values(
            [self.user_col, self.timestamp_col], kind="stable")
        
        data["time_idx_reversed"] = data.groupby(self.user_col).cumcount(
            ascending=False
        )

        train = data[data.time_idx_reversed >= 2].drop(columns=["time_idx_reversed"])
        validation = data[data.time_idx_reversed >= 1].drop(
            columns=["time_idx_reversed"]
        )
        test = data.drop(columns=["time_idx_reversed"])

        return train, validation, test


class GlobalTimeSplitter:
    """Global temporal split."""

    def __init__(
        self,
        quantile,
        validation_quantile=0.9,
        validation_type="by_user",
        validation_size=500,
        user_col="user_id",
        timestamp_col="timestamp",
        random_state=42,
    ):
        self.quantile = quantile
        self.validation_quantile = validation_quantile
        self.validation_type = validation_type
        self.validation_size = validation_size
        self.user_col = user_col
        self.timestamp_col = timestamp_col
        self.random_state = random_state

    def split(self, data):

        val_time_threshold = None
        train, test, time_threshold = self.split_by_time(data, self.quantile)

        if self.validation_type == "by_user":
            train, validation = self.split_validation_by_user(train)
        elif self.validation_type == "by_time":
            train, validation, val_time_threshold = self.split_by_time(train, self.validation_quantile)
        elif self.validation_type == "last_train_item":
            train, validation = self.split_validation_last_train(train)
        else:
            raise ValueError("Wrong validation_type.")

        return train, validation, test, time_threshold, val_time_threshold

    def split_by_time(self, data, quantile):
        data = data.sort_values(
            [self.user_col, self.timestamp_col], kind="stable")
        
        time_threshold = data[self.timestamp_col].quantile(quantile)

        # we need at least two items in a train sequence for training
        user_second_timestamp = data.groupby("user_id")[self.timestamp_col].nth(1)
        train_users = user_second_timestamp[
            user_second_timestamp <= time_threshold
        ].index
        train = data[data[self.user_col].isin(train_users)]
        # train contains all interactions before the time threshold
        train = train[train[self.timestamp_col] <= time_threshold]

        # test contains users with the last interaction after the time threshold
        user_last_timestamp = data.groupby("user_id")[self.timestamp_col].nth(-1)
        test_users = user_last_timestamp[user_last_timestamp > time_threshold].index
        test = data[data[self.user_col].isin(test_users)]

        return train, test, time_threshold

    def split_validation_by_user(self, train):
        if self.validation_size is None:
            raise ValueError("You must specify split_params.validation_size parameter for by_user splitting")
        np.random.seed(self.random_state)
        validation_users = np.random.choice(
            train[self.user_col].unique(), size=self.validation_size, replace=False
        )
        validation = train[train[self.user_col].isin(validation_users)]
        train = train[~train[self.user_col].isin(validation_users)]

        return train, validation
    
    def split_validation_last_train(self, train):
        train = train.sort_values(
            [self.user_col, self.timestamp_col], kind="stable")
        train["time_idx_reversed"] = train.groupby(self.user_col).cumcount(ascending=False)
        
        # at least two in validation
        validation = train[train.groupby(self.user_col)['time_idx_reversed'].transform(max) > 0].drop(
            columns=["time_idx_reversed"]
        )
        
        train = train[train.time_idx_reversed >= 1]
        # at least two in train
        train = train[train.groupby(self.user_col)['time_idx_reversed'].transform(max) > 1].drop(
            columns=["time_idx_reversed"]
        )

        return train, validation


