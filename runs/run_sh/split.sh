export SEQ_SPLITS_DATA_PATH=$(pwd)/data
export PYTHONPATH='./'

set -e

# LLO
python runs/split.py -m split_type=leave-one-out dataset=Beauty,BeerAdvocate,Diginetica,Movielens-1m,Sports,Zvuk,Movielens-20m,YooChoose

# GT by_user
python runs/split.py -m split_type=global_timesplit split_params.validation_size=1024 split_params.quantile=0.9 split_params.validation_type=by_user dataset=Beauty,BeerAdvocate,Diginetica,Movielens-1m,Sports,Zvuk,Movielens-20m,YooChoose

# GT last_train_item by_time
python runs/split.py -m split_type=global_timesplit split_params.quantile=0.9 split_params.validation_type=last_train_item dataset=Beauty,BeerAdvocate,Diginetica,Movielens-1m,Sports,Zvuk,Movielens-20m,YooChoose

# GT by_time
python runs/split.py -m split_type=global_timesplit split_params.quantile=0.8,0.9,0.95,0.975 split_params.validation_type=by_time dataset=Beauty,BeerAdvocate,Diginetica,Movielens-1m,Sports,Zvuk,Movielens-20m,YooChoose 
