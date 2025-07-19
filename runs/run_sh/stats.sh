#!/bin/bash

export SEQ_SPLITS_DATA_PATH=$(pwd)/data
export PYTHONPATH='./'

set -e

python runs/statistics.py -m split_type=raw,preprocessed dataset=Beauty,BeerAdvocate,Diginetica,Movielens-1m,Sports,Zvuk,Movielens-20m,YooChoose

python runs/statistics.py -m split_type=leave-one-out dataset=Beauty,BeerAdvocate,Diginetica,Movielens-1m,Sports,Zvuk,Movielens-20m,YooChoose

python runs/statistics.py -m split_type=global_timesplit split_params.quantile=0.9 split_params.validation_type=val_by_user,val_by_time,val_last_train_item dataset=Beauty,BeerAdvocate,Diginetica,Movielens-1m,Sports,Zvuk,Movielens-20m,YooChoose

python runs/statistics.py -m split_type=global_timesplit split_params.quantile=0.8,0.95,0.975 split_params.validation_type=val_by_time dataset=Beauty,BeerAdvocate,Diginetica,Movielens-1m,Sports,Zvuk,Movielens-20m,YooChoose
