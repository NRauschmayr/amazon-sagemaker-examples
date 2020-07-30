#  Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License").
#  You may not use this file except in compliance with the License.
#  A copy of the License is located at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  or in the "license" file accompanying this file. This file is distributed
#  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#  express or implied. See the License for the specific language governing
#  permissions and limitations under the License.
from __future__ import print_function

import argparse
import json
import logging
import os
import pandas as pd
import pickle as pkl

from sagemaker_containers import entry_point
from sagemaker_xgboost_container.data_utils import get_dmatrix
from smdebug.xgboost import Hook, SaveConfig

import xgboost as xgb



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here.
    parser.add_argument('--max_depth', type=int,)
    parser.add_argument('--eta', type=float)
    parser.add_argument('--gamma', type=int)
    parser.add_argument('--min_child_weight', type=int)
    parser.add_argument('--subsample', type=float)
    parser.add_argument('--objective', type=str)
    parser.add_argument('--num_round', type=int)

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--s3_bucket', type=str, default=None)
    
    args, _ = parser.parse_known_args()

    dtrain = get_dmatrix(args.train, 'libsvm')
    dval = get_dmatrix(args.validation, 'libsvm')
    watchlist = [(dtrain, 'train'), (dval, 'validation')] if dval is not None else [(dtrain, 'train')]

    job_name = json.loads(os.environ['SM_TRAINING_ENV'])['job_name']
    path = 's3://' + args.s3_bucket + "/" + job_name + "/debug-output"
    
    train_hp = {
        'max_depth': args.max_depth,
        'eta': args.eta,
        'gamma': args.gamma,
        'min_child_weight': args.min_child_weight,
        'subsample': args.subsample,
        'objective': args.objective
        }
    hook = Hook(
        out_dir=path,  
        include_collections=['feature_importance', 'full_shap', 'average_shap', 'labels', 'predictions'],
        train_data=dtrain,
        validation_data=dval,
        hyperparameters=train_hp,
        save_config=SaveConfig(save_interval=10)
    )
    
    booster = xgb.train(params=train_hp,
                        dtrain=dtrain,
                        num_boost_round=args.num_round,
                        evals=watchlist,
                        callbacks=[hook])
    