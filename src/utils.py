""""
Utils.
"""

import os
from glob import glob

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def extract_validation_history(path):
    """Extract validation history from lightning logs."""

    events_path = glob(os.path.join(path, 'events.*'))[0]

    event_acc = EventAccumulator(events_path)
    event_acc.Reload()

    scalars = event_acc.Tags()['scalars']
    history = pd.DataFrame(columns=['step'])
    for scalar in scalars:
        events = event_acc.Scalars(tag=scalar)
        df_scalar = pd.DataFrame(
            [(event.step, event.value) for event in events], columns=['step', scalar])
        history = pd.merge(history, df_scalar, on='step', how='outer')

    if 'epoch' in scalars:
        events = event_acc.Scalars(tag='epoch')
        df_time = pd.DataFrame(
            [(event.step, event.wall_time) for event in events], columns=['step', 'time'])
        df_time.time = df_time.time - df_time.time.loc[0]
        history = pd.merge(history, df_time, on='step', how='outer')

    return history
