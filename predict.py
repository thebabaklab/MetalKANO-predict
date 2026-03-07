# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 19:42:36 2019

@author: SY
"""
import os
import warnings

import pandas as pd
from rdkit import RDLogger

from chemprop.parsing import modify_train_args, parse_train_args
from chemprop.train import make_predictions

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")


if __name__ == "__main__":
    args = parse_train_args()
    if not hasattr(args, 'checkpoint_path') or not args.checkpoint_path:
        raise SystemExit("Error: --checkpoint_path is required (path to a .pt model file)")
    args.num_tasks = 1
    args.dataset_type = "classification"
    modify_train_args(args)

    data = pd.read_csv(args.data_path)
    pred, smiles = make_predictions(args, data.smiles.tolist())

    df = pd.DataFrame({"smiles": smiles})
    for i in range(len(pred[0])):
        df[f"pred_{i}"] = [item[i] for item in pred]
    
    # Create output directory if it doesn't exist
    output_dir = f"./{args.exp_name}/{args.exp_id}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/predict.csv"
    
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
