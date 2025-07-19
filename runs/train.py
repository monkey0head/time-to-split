"""
Run experiment.
"""

import os
import time

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from clearml import Task
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, ModelCheckpoint,
                                         ModelSummary, TQDMProgressBar)
from torch.utils.data import DataLoader

from src.datasets import (CausalLMDataset, CausalLMPredictionDataset, SuccessivePredictionDataset,
                          MaskedLMDataset, MaskedLMPredictionDataset,
                          PaddingCollateFn)
from src.metrics import Evaluator
from src.models import SASRec, GRU4Rec, BERT4Rec
from src.modules import SeqRec
from src.postprocess import preds2recs
from src.prepr import last_item_split, global_temporal_split, get_ground_truth_from_successive

import itertools
import pickle
from tqdm.auto import tqdm


def get_grid_config(index, hyper, model_class):

    if model_class == "GRU4Rec":
        if index is None:
            return OmegaConf.create({
                "hidden_size": 64,
                "num_layers": 1,
                "dropout": 0.1
            })
        grid = list(itertools.product(hyper.hidden_size, hyper.num_layers, hyper.dropout))
        if index < 0 or index >= len(grid):
            raise IndexError(f"Index {index} is out of range. Must be between 0 and {len(grid) - 1}")
        selected = grid[index]
        return OmegaConf.create({
            "hidden_size": selected[0],
            "num_layers": selected[1],
            "dropout": selected[2]
        })
    elif model_class == "SASRec":
        if index is None:
            return OmegaConf.create({
                "hidden_units": 64,
                "num_blocks": 2,
                "num_heads": 2,
                "dropout_rate": 0.1
            })
        grid = list(itertools.product(hyper.hidden_units, hyper.num_blocks, hyper.num_heads, hyper.dropout_rate))
        if index < 0 or index >= len(grid):
            raise IndexError(f"Index {index} is out of range. Must be between 0 and {len(grid) - 1}")
        selected = grid[index]
        return OmegaConf.create({
            "hidden_units": selected[0],
            "num_blocks": selected[1],
            "num_heads": selected[2],
            "dropout_rate": selected[3]
        })
    elif model_class == "BERT4Rec":
        if index is None:
            return OmegaConf.create({
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "hidden_dropout_prob": 0.1
            })
        grid = list(itertools.product(hyper.hidden_size, hyper.num_hidden_layers, hyper.num_attention_heads, hyper.hidden_dropout_prob))
        if index < 0 or index >= len(grid):
            raise IndexError(f"Index {index} is out of range. Must be between 0 and {len(grid) - 1}")
        selected = grid[index]
        return OmegaConf.create({
            "hidden_size": selected[0],
            "num_hidden_layers": selected[1],
            "num_attention_heads": selected[2],
            "hidden_dropout_prob": selected[3]
        })
    else:
        raise NotImplementedError(f"Grid configuration not implemented for model class: {model_class}")
OmegaConf.register_new_resolver("get_grid_config", get_grid_config)
OmegaConf.register_new_resolver("calc", lambda expr: eval(expr, {}, {}))


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(config):

    print(OmegaConf.to_yaml(config, resolve=True))

    if hasattr(config, 'cuda_visible_devices'):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.cuda_visible_devices)

    if config.clearml_project_folder is not None:
        split_subtype = config.split_subtype or ''
        project_name = os.path.join(
            config.clearml_project_folder, config.split_type, split_subtype,
            config.dataset.name, config.model.model_class)
        task = Task.init(project_name=project_name, task_name=config.clearml_task_name,
                        reuse_last_task_id=False)
        task.connect(OmegaConf.to_container(config, resolve=True))
    else:
        task = None

    fix_seeds(config.random_state)
    
    train, validation, test, max_item_id, global_timepoint, global_timepoint_val = prepare_data(config)
    
    train_loader, eval_loader = create_dataloaders(train, validation, config)
    model = create_model(config, item_count=max_item_id)
    
    trainer, seqrec_module, num_best_epoches = training(model, train_loader, eval_loader, config, task)
        
    if trainer is not None:
        recs_validation = run_eval(validation, trainer, seqrec_module, config, max_item_id, task, global_timepoint_val, prefix='val')
    else:
        print('Skipping validation') 

    test_prefix = 'test'
    if config.retrain_with_validation:
        merged = pd.concat([train, validation], ignore_index=True)
        merged_loader, _ = create_dataloaders(merged, pd.DataFrame([], columns=validation.columns), config)
        model = create_model(config, item_count=max_item_id)
        config.trainer_params.max_epochs = num_best_epoches
        trainer, seqrec_module, _ = training(model, merged_loader, None, config, task, retrain=True)
        test_prefix = 'test_retrained'
    
    recs_test = run_eval(test, trainer, seqrec_module, config, max_item_id, task, global_timepoint, prefix=test_prefix)


def fix_seeds(random_state):
    """Set up random seeds."""

    seed_everything(random_state, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_data(config):
    
    split_subtype = config.split_subtype or ''
    split_type = config.split_type
    if split_type == 'global_timesplit':
        if config.quantile is not None:
            q = 'q0'+str(config.quantile)[2:]
        else:
            raise ValueError("'global_timesplit' split must be run with parameter 'quantile'")
    else:
        q = ''
    data_path = os.path.join(
        os.environ["SEQ_SPLITS_DATA_PATH"], 'splitted', split_type,
        split_subtype, config.dataset.name, q)

    train = pd.read_csv(os.path.join(data_path, 'train.csv'))
    print('train shape', train.shape)
    validation = pd.read_csv(os.path.join(data_path, 'validation.csv'))
    print('validation shape', validation.shape)
    test = pd.read_csv(os.path.join(data_path, 'test.csv'))
    print('test shape', test.shape)

    # index 1 is used for masking value
    if config.model.model_class == 'BERT4Rec':
        train.item_id += 1
        validation.item_id += 1
        test.item_id += 1

    max_item_id = max(train.item_id.max(), test.item_id.max(), validation.item_id.max())

    global_timepoint = None
    global_timepoint_val = None
    if split_type == 'global_timesplit':
        with open(os.path.join(data_path,'time_threshold.pkl'), 'rb') as f:
            global_timepoint = pickle.load(f)
        print('Test global timepoint', global_timepoint)
        if split_subtype == 'val_by_time':
            with open(os.path.join(data_path,'val_time_threshold.pkl'), 'rb') as f:
                global_timepoint_val = pickle.load(f)
            print('Validation global timepoint', global_timepoint_val)
    print()
    return train, validation, test, max_item_id, global_timepoint, global_timepoint_val


def create_dataloaders(train, validation, config):

    validation_size = config.dataloader.validation_size
    validation_users = validation.user_id.unique()
    if validation_size and (validation_size < len(validation_users)):
        validation_users = np.random.choice(validation_users, size=validation_size, replace=False)
        validation = validation[validation.user_id.isin(validation_users)]

    train_dataset = (
        MaskedLMDataset(train, mlm_probability=config.model.mlm_probability, **config['dataset_params'])
        if config.model.model_class == 'BERT4Rec'
        else CausalLMDataset(train, **config['dataset_params'])
    )
    eval_dataset = (
        MaskedLMPredictionDataset(validation, max_length=config.dataset_params.max_length, validation_mode=True)
        if config.model.model_class == 'BERT4Rec'
        else CausalLMPredictionDataset(validation, max_length=config.dataset_params.max_length, validation_mode=True)
    )

    train_loader = DataLoader(train_dataset, batch_size=config.dataloader.batch_size,
                              shuffle=True, num_workers=config.dataloader.num_workers,
                              collate_fn=PaddingCollateFn())
    eval_loader = DataLoader(eval_dataset, batch_size=config.dataloader.test_batch_size,
                             shuffle=False, num_workers=config.dataloader.num_workers,
                             collate_fn=PaddingCollateFn())

    return train_loader, eval_loader


def create_model(config, item_count):

    if config.model.model_class == 'SASRec':
        model = SASRec(item_num=item_count, **config.model.model_params)
    elif config.model.model_class == 'GRU4Rec':
        model = GRU4Rec(vocab_size=item_count + 1,
                    rnn_config=config.model.model_params)
    elif config.model.model_class == 'BERT4Rec':
        model = BERT4Rec(vocab_size=item_count + 1, # ok since we add 1 to item_id
                         add_head=True,
                         tie_weights=True,
                         bert_config=config.model.model_params)
    return model


def training(model, train_loader, eval_loader, config, task=None, retrain=False):

    split_subtype = config.split_subtype or ''
    grid_point_number = config.model.grid_point_number if config.model.grid_point_number is not None else 'X'
    q = 'q0' + str(config.quantile)[2:] if config.split_type == 'global_timesplit' else ''
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', config.split_type,
        split_subtype, config.dataset.name, q, config.model.model_class, 'retrain_with_val' if retrain else '')
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if config.model.model_class == 'SASRec':
        file_name = (
            f"{grid_point_number}_"
            f"{config.model.model_params.hidden_units}_"
            f"{config.model.model_params.num_blocks}_"
            f"{config.model.model_params.num_heads}_"
            f"{config.model.model_params.dropout_rate}_"
            f"{config.model.model_params.maxlen}_"
            f"{config.dataloader.batch_size}_"
            f"{config.random_state}"
        )
    elif config.model.model_class == 'GRU4Rec':
        file_name = (
            f"{grid_point_number}_"
            f"{config.model.model_params.hidden_size}_"
            f"{config.model.model_params.num_layers}_"
            f"{config.model.model_params.dropout}_"
            f"{config.model.model_params.input_size}_"
            f"{config.dataloader.batch_size}_"
            f"{config.random_state}"
        )
    elif config.model.model_class == 'BERT4Rec':
        file_name = (
            f"{grid_point_number}_"
            f"{config.model.model_params.hidden_size}_"
            f"{config.model.model_params.num_hidden_layers}_"
            f"{config.model.model_params.num_attention_heads}_"
            f"{config.model.model_params.hidden_dropout_prob}_"
            f"{config.dataloader.batch_size}_"
            f"{config.random_state}"
        )
    checkpoint_file = os.path.join(model_path, file_name + ".ckpt")

    seqrec_module = SeqRec(model, **config['seqrec_module']) 
    if getattr(config, 'load_if_possible', False) and os.path.exists(checkpoint_file):
        checkpoint_dict = torch.load(checkpoint_file)
        seqrec_module.load_state_dict(checkpoint_dict['state_dict'])
        num_best_epoches = checkpoint_dict['epoch'] + 1
        print(f'Loaded trained model from: {checkpoint_file}')
        return None, seqrec_module, num_best_epoches
    
    model_summary = ModelSummary(max_depth=1)
    progress_bar = TQDMProgressBar(refresh_rate=20)
    if not retrain:
        checkpoint = ModelCheckpoint(
            dirpath=model_path,  
            filename='_' + file_name,           
            save_top_k=1,
            monitor="val_ndcg",
            mode="max",
            save_weights_only=True
        )
        early_stopping = EarlyStopping(monitor="val_ndcg", mode="max",
                                    patience=config.patience, verbose=False)
        callbacks = [early_stopping, model_summary, checkpoint, progress_bar]
    else:
        checkpoint = ModelCheckpoint(
            dirpath=model_path,  
            filename='_' + file_name,           
            save_weights_only=True
        )
        callbacks = [model_summary, checkpoint, progress_bar]

    trainer = pl.Trainer(callbacks=callbacks, enable_checkpointing=True,
                         **config['trainer_params'])
    
    start_time = time.time()
    try:
        trainer.fit(model=seqrec_module,
                    train_dataloaders=train_loader,
                    val_dataloaders=eval_loader)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            if os.path.exists(checkpoint.best_model_path):
                os.remove(checkpoint.best_model_path)
                print(f"Removed checkpoint due to CUDA OOM error: {checkpoint.best_model_path}")
        raise
    finally:
        if not getattr(trainer, "interrupted", False):
            seqrec_module.load_state_dict(torch.load(checkpoint.best_model_path)['state_dict'])
            os.rename(checkpoint.best_model_path, checkpoint_file)
            print(f"Checkpoint renamed to: {checkpoint_file}")
            num_best_epoches = torch.load(checkpoint_file)['epoch'] + 1
        else:
            print("Detected interruption of training.")
            os.remove(checkpoint.best_model_path)

    training_time = time.time() - start_time
    print('training_time', training_time)

    if task is not None:
        task.get_logger().report_single_value('training_time', training_time)
        task.close()

    return trainer, seqrec_module, num_best_epoches


def predict(trainer, seqrec_module, data, config, global_timepoint, successive):

    if not successive:
        if config.model.model_class == 'SASRec' or config.model.model_class == 'GRU4Rec':
            predict_dataset = CausalLMPredictionDataset(data, max_length=config.dataset_params.max_length)
        elif config.model.model_class == 'BERT4Rec':
            predict_dataset = MaskedLMPredictionDataset(data, max_length=config.dataset_params.max_length)
    else:
        if config.model.model_class == 'SASRec' or config.model.model_class == 'GRU4Rec':
            predict_dataset = SuccessivePredictionDataset(data, global_timepoint, max_length=config.dataset_params.max_length)
        elif config.model.model_class == 'BERT4Rec':
            predict_dataset = SuccessivePredictionDataset(data, global_timepoint, max_length=config.dataset_params.max_length,
                                                          masking_value=config.model.masking_value)
    predict_loader = DataLoader(
        predict_dataset, shuffle=False,
        collate_fn=PaddingCollateFn(),
        batch_size=config.dataloader.test_batch_size, 
        num_workers=config.dataloader.num_workers)

    seqrec_module.predict_top_k = max(config.evaluator.top_k)

    if trainer is None:
        predict_trainer_params = config.get('trainer_predict_params', {})
        trainer = pl.Trainer(callbacks=[TQDMProgressBar(refresh_rate=100)], **predict_trainer_params)

    preds = trainer.predict(model=seqrec_module, dataloaders=predict_loader)
    recs = preds2recs(preds, successive=successive)
    print('recs shape', recs.shape)

    return recs


def run_eval(data, trainer, seqrec_module, config, max_item_id, task, global_timepoint, prefix):
    """Get predictions and ground truth for selected ground truth type. Calculate metrics."""
    start_time = time.time()
    # TO DO: modify if for
    # - llo
    # - gts first, last, all, successive
    # - all items after gts
    test_ground_truth = None
    recs = None
    # for now for global_timesplit if not successive, use last item as gt
    if config.split_type == "leave-one-out" or not config.evaluator[f"successive_{prefix}"]:
        test, test_ground_truth = last_item_split(data)
        recs = predict(trainer, seqrec_module, test, config, global_timepoint, successive=False)
        if config.split_type != 'leave-one-out':
            prefix = f'{prefix}_last'
        evaluate(recs, test_ground_truth, task, config, max_item_id, prefix=prefix, successive=False)

    # elif config.split_type == "global_timesplit" and config.evaluator.ground_truth_type == "all_after_timepoint":

    else:
        # successive
        recs = predict(trainer, seqrec_module, data, config, global_timepoint, successive=True)
        if config.evaluator[f"calc_successive_metrics_{prefix}"]:
            evaluate(recs, test_ground_truth, task, config, max_item_id, prefix=prefix, successive=True)

        # different ground truth positions from successive
        different_gt_and_recs = {}
        recs["gt_position"] = recs["pseudo_user_id"] - recs.groupby(["user_id"])["pseudo_user_id"].transform(min)
        
        # first_item
        different_gt_and_recs["first"] = recs[recs["gt_position"] == 0]

        # last_item
        different_gt_and_recs["last"] = recs[
            recs["gt_position"] == recs.groupby("user_id")["gt_position"].transform(max)
        ]

        # random_item
        np.random.seed(config.random_state)
        random_gt_position = recs.groupby("user_id")["gt_position"].max().apply(lambda x: np.random.randint(0, x + 1))
        different_gt_and_recs["random"] = recs[recs["gt_position"] == recs["user_id"].map(random_gt_position)]
        
        ## all items in gt from successive recs
        evaluate(different_gt_and_recs["first"], get_ground_truth_from_successive(recs), 
                     task, config, max_item_id, prefix=f"{prefix}_all_items_s", successive=False)

        ## all items in gt from successive recs for warm users only
        warm_users = data[data['timestamp'] <= global_timepoint]["user_id"].unique()
        evaluate(different_gt_and_recs["first"][different_gt_and_recs["first"]["user_id"].isin(warm_users)], 
                 get_ground_truth_from_successive(recs[recs["user_id"].isin(warm_users)]), 
                     task, config, max_item_id, prefix=f"{prefix}_all_items_s_warm", successive=False)

        ## all items no successive to make separate functionality afterwards
        # test, test_ground_truth = global_temporal_split(data, global_timepoint)
        # recs = predict(trainer, seqrec_module, test, config, global_timepoint, successive=False)
        # evaluate(recs, test_ground_truth, 
        #              task, config, max_item_id, prefix=f"{prefix}_all_items", successive=False)

        for gt_name, current_recs in different_gt_and_recs.items():
            evaluate(current_recs, get_ground_truth_from_successive(current_recs), 
                     task, config, max_item_id, prefix=f"{prefix}_{gt_name}", successive=False,
            )

    eval_time = time.time() - start_time
    print(f"{prefix} predict and evaluation time", eval_time)

    if task is not None:
        task.get_logger().report_single_value(f"{prefix}_time", eval_time)
        if config[f"save_{prefix}_predictions"]:
            task.upload_artifact(f"{prefix}_pred.csv", recs)

    return recs

def evaluate(recs, ground_truth, task, config, num_items=None, prefix='test', successive=False):    
    if not successive:
        all_items = None
        if "Coverage" in config.evaluator.metrics:
            all_items = pd.DataFrame({"item_id": np.arange(1, num_items + 1)})    
            # to pass replay data check
            all_items["user_id"] = 0

        evaluator = Evaluator(metrics=list(config.evaluator.metrics),
                            top_k=list(config.evaluator.top_k))
        metrics = evaluator.compute_metrics(ground_truth, recs, train=all_items)
        print(f'{prefix} metrics:\n', metrics_dict_to_df(metrics, config), '\n')
        metrics = {prefix + '_' + key: value for key, value in metrics.items()}

        if task:
            clearml_logger = task.get_logger()
            for key, value in metrics.items():
                clearml_logger.report_single_value(key, value)
            metrics = pd.Series(metrics).to_frame().reset_index()
            metrics.columns = ['metric_name', 'metric_value']

            clearml_logger.report_table(title=f'{prefix}_metrics', series='dataframe',
                                        table_plot=metrics)
            task.upload_artifact(f'{prefix}_metrics', metrics)
        else:
            metrics = pd.Series(metrics).to_frame().reset_index()
            metrics.columns = ['metric_name', 'metric_value']
            save_local_metrics(metrics, config, prefix)
    else:
        replay_metrics = config.evaluator.successive_replay_metrics
        final_metrics_global, final_metrics_user = evaluate_successive_replay(recs, num_items, config) if replay_metrics else evaluate_successive(recs, num_items, config)

        print("Successive Global Metrics:")
        print(final_metrics_global, '\n')
        print("Successive User Average Metrics:")
        print(final_metrics_user, '\n')
            
        combined_df = pd.concat([final_metrics_global, final_metrics_user], axis=0, keys=["global", "user"])
        combined_df = combined_df.reset_index(level=0).rename(columns={"level_0": "evaluation_type"})
        save_local_metrics(combined_df, config, f'{prefix}_successive')

def evaluate_successive(recs, num_items, config):
    
    global_results = {}
    user_avg_results = {}
    
    for k in tqdm(config.evaluator.top_k, desc="Processing @K"):
        cum_hits = 0
        cum_reciprocal_ranks = 0.
        cum_discounts = 0.
        global_unique_recs = set()
        total_count = 0

        per_user_hr = []
        per_user_rr = []
        per_user_ndcg = []
        per_user_cov = []
        
        for _, user in recs.groupby('user_id'):

            test_seq = user.drop_duplicates('pseudo_user_id')['target']
            num_predictions = len(test_seq)
            if not num_predictions:                               # if no test items left - skip user
                continue
            predicted_items = user['item_id'].values.reshape(-1, max(config.evaluator.top_k))[:, :k]

            user_unique_recs = set(predicted_items.ravel())
            
            _, hit_index = np.where(predicted_items == np.atleast_2d(test_seq).T)

            user_rr_sum = 0.0
            user_ndcg_sum = 0.0
            num_hits = hit_index.size

            if num_hits:
                cum_hits += num_hits

                user_rr_sum  = np.sum(1. / (hit_index + 1))
                cum_reciprocal_ranks += user_rr_sum
                
                user_ndcg_sum =  np.sum(1. / np.log2(hit_index + 2))
                cum_discounts += user_ndcg_sum
            total_count += num_predictions
            global_unique_recs.update(user_unique_recs)
            
            per_user_hr.append(num_hits / num_predictions)
            per_user_rr.append(user_rr_sum / num_predictions)
            per_user_ndcg.append(user_ndcg_sum / num_predictions)
            per_user_cov.append(len(user_unique_recs) / num_items)

        global_hr = cum_hits / total_count if total_count > 0 else 0
        global_mrr = cum_reciprocal_ranks / total_count if total_count > 0 else 0
        global_ndcg = cum_discounts / total_count if total_count > 0 else 0
        global_cov = len(global_unique_recs) / num_items

        global_results[f'@{k}'] = {'NDCG': global_ndcg, 'HR': global_hr, 'MRR': global_mrr, 'COV': global_cov}

        avg_hr = np.mean(per_user_hr)
        avg_mrr = np.mean(per_user_rr)
        avg_ndcg = np.mean(per_user_ndcg)
        avg_cov = np.mean(per_user_cov)
        
        user_avg_results[f'@{k}'] = {'NDCG': avg_ndcg, 'HR': avg_hr, 'MRR': avg_mrr, 'COV': avg_cov}

    metrics = ['NDCG', 'HR', 'MRR', 'COV']
    final_metrics_global = pd.DataFrame(index=metrics, columns=[f'@{k}' for k in config.evaluator.top_k])
    final_metrics_user = pd.DataFrame(index=metrics, columns=[f'@{k}' for k in config.evaluator.top_k])
    
    for col in final_metrics_global.columns:
        for metric in metrics:
            final_metrics_global.loc[metric, col] = global_results[col][metric]
            final_metrics_user.loc[metric, col] = user_avg_results[col][metric]

    return final_metrics_global, final_metrics_user

def evaluate_successive_replay(recs, num_items, config):
    
    ground_truth = recs[['user_id', 'pseudo_user_id', 'target']].drop_duplicates().rename(columns={'target':'item_id'})
    
    all_items = None
    if "Coverage" in config.evaluator.metrics:
        all_items = pd.DataFrame({"item_id": np.arange(1, num_items + 1)})
        all_items['pseudo_user_id'] = 0

    evaluator = Evaluator(metrics=list(config.evaluator.metrics),
                          top_k=list(config.evaluator.top_k), user_col='pseudo_user_id')
    print('Processing Global Metrics ...')
    final_metrics_global = evaluator.compute_metrics(ground_truth, recs, train=all_items)

    final_metrics_user = []
    for _, user in tqdm(recs.groupby('user_id'), desc='Processing Userbased Metrics'):
        final_metrics_user.append(evaluator.compute_metrics(ground_truth[ground_truth['user_id'] == user['user_id'].iloc[0]], 
                                                            recs[recs['user_id'] == user['user_id'].iloc[0]], 
                                                            train=all_items)
                                                            )
    final_metrics_user = {key: sum(d[key] for d in final_metrics_user) / len(final_metrics_user) for key in final_metrics_user[0]}

    final_metrics_global = metrics_dict_to_df(final_metrics_global, config)
    final_metrics_user = metrics_dict_to_df(final_metrics_user, config)
    
    return final_metrics_global, final_metrics_user

def metrics_dict_to_df(d, config):

    metrics = config.evaluator.metrics
    ks = config.evaluator.top_k
    data = {
        metric: [
            d.get(f"{'HitRate' if metric=='HR' else metric}@{k}", float('nan'))
            for k in ks
        ]
        for metric in metrics
    }
    df = pd.DataFrame(data, index=[f"@{k}" for k in ks]).T.reindex(metrics)
    
    return df.rename(index={"HitRate": "HR", "Coverage": "COV"})

def save_local_metrics(metrics, config, prefix):
    
    split_subtype = config.split_subtype or ''
    q = 'q0'+str(config.quantile)[2:] if config.split_type == 'global_timesplit' else ''
    grid_point_number = config.model.grid_point_number if config.model.grid_point_number is not None else 'X'
    results_path = os.path.join(
        os.environ["SEQ_SPLITS_DATA_PATH"], 'results', config.split_type,
        split_subtype, config.dataset.name, q, config.model.model_class, prefix)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    if config.model.model_class == 'SASRec':
        file_name = (
            f"{grid_point_number}_"
            f"{config.model.model_params.hidden_units}_"
            f"{config.model.model_params.num_blocks}_"
            f"{config.model.model_params.num_heads}_"
            f"{config.model.model_params.dropout_rate}_"
            f"{config.model.model_params.maxlen}_"
            f"{config.dataloader.batch_size}_"
            f"{config.random_state}.csv"
        )
    elif config.model.model_class == 'GRU4Rec':
        file_name = (
            f"{grid_point_number}_"
            f"{config.model.model_params.hidden_size}_"
            f"{config.model.model_params.num_layers}_"
            f"{config.model.model_params.dropout}_"
            f"{config.model.model_params.input_size}_"
            f"{config.dataloader.batch_size}_"
            f"{config.random_state}.csv"
        )
    elif config.model.model_class == 'BERT4Rec':
        file_name = (
            f"{grid_point_number}_"
            f"{config.model.model_params.hidden_size}_"
            f"{config.model.model_params.num_hidden_layers}_"
            f"{config.model.model_params.num_attention_heads}_"
            f"{config.model.model_params.hidden_dropout_prob}_"
            f"{config.dataloader.batch_size}_"
            f"{config.random_state}.csv"
        )

    metrics.to_csv(results_path + '/' + file_name, index=True)


if __name__ == "__main__":

    main()
