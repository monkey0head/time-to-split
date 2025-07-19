"""
Postprocessing.
"""

import numpy as np
import pandas as pd


def preds2recs(preds, item_mapping=None, successive=False):

    user_ids = np.hstack([pred['user_ids'] for pred in preds])
    scores = np.vstack([pred['scores'] for pred in preds])
    preds_arr = np.vstack([pred['preds'] for pred in preds])
    
    user_ids_rep = np.repeat(user_ids[:, None], repeats=scores.shape[1], axis=1)
    
    recs = pd.DataFrame({
        'user_id': user_ids_rep.flatten(),
        'item_id': preds_arr.flatten(),
        'prediction': scores.flatten()
    })
    
    if successive:
        top_k = preds[0]['scores'].shape[1]
        total_users = sum(pred['user_ids'].shape[0] for pred in preds)
        recs['pseudo_user_id'] = np.repeat(np.arange(total_users), repeats=top_k)
        recs['target'] = np.hstack([
            np.repeat(pred['target_ids'], repeats=top_k)
            for pred in preds
        ])

    if item_mapping is not None:
        recs['item_id'] = recs['item_id'].map(item_mapping)
    
    return recs