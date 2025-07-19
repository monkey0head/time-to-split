"""
Make user and item indexes.
"""

import pandas as pd


def encode(data, col_name, shift):
    """Encode items/users to consecutive ids.

    :param col_name: column to do label encoding, e.g. 'item_id'
    :param shift: shift encoded values to start from shift
    """
    data[col_name] = data[col_name].astype("category").cat.codes + shift
    return data
