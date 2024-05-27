from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics
from scipy.stats import pearsonr
from transformers import BatchFeature
import torch
from datasets import Dataset
import numpy as np
import pandas as pd
import os, pickle, glob
import argparse
import sys

import sys
sys.path.append(r'/home/chendo11/workfolder/TopTransformer/code_pkg')
from top_transformer import TopTForImageClassification
from top_transformer import TopTForPreTraining
from top_transformer import TopTConfig



from PL_transformer import PLFForImageClassification
from PL_transformer import PLFForPreTraining
from PL_transformer import PLFConfig


def metrics_func(true_value, predict_value):
    # metrics
    r2 = metrics.r2_score(true_value, predict_value)
    mae = metrics.mean_absolute_error(true_value, predict_value)
    mse = metrics.mean_squared_error(true_value, predict_value)
    rmse = mse ** 0.5
    pearson_r = pearsonr(true_value, predict_value)[0]
    pearson_r2 = pearson_r ** 2

    # print
    print(f"Metric - r2: {r2:.3f} mae: {mae:.3f} mse: {mse:.3f} "
          f"rmse: {rmse:.3f} pearsonr: {pearson_r:.3f} pearsonr2: {pearson_r2:.3f}")
    return r2, mae, mse, rmse, pearson_r, pearson_r2


def load_and_save_pretrained():

    original_model_path = r'/home/chendo11/workfolder/Pfizer_project/PLFormer_output_models/pretrained_model'


    # load model
    config = TopTConfig.from_pretrained(original_model_path)
    model = TopTForPreTraining.from_pretrained(original_model_path, config=config)
    # print(config)

    original_state_dict = model.state_dict()

    new_state_dict = {}
    for key, value in original_state_dict.items():
        # print(key)
        new_key = key.replace('topt', 'plf')  # Replace 'topt' with 'plf'
        # print(new_key)
        new_state_dict[new_key] = value

    # creat the plf initial model
    plf_config_dict = {}
    for k, v in config.to_dict().items():
        if k == 'architectures':
            v = 'PLFForPreTraining'
        elif k == 'model_type':
            v = 'plf'
        else:
            plf_config_dict[k] = v
    plf_config = PLFConfig()
    plf_config.update(plf_config_dict)
    plf_model = PLFForPreTraining(plf_config)
    # print(plf_model)

    # reset the parameters
    plf_model.load_state_dict(new_state_dict)

    # # save the model
    save_directory = r'/home/chendo11/workfolder/Pfizer_project/PLFormer_output_models/pretrained_plf_model' 
    plf_model.save_pretrained(save_directory)

    # # test the saved model
    config_test = PLFConfig.from_pretrained(save_directory)
    model = PLFForPreTraining.from_pretrained(save_directory, config=config)
    print(config_test)

    return None


def load_and_save_finetuned():

    original_model_path = r'/home/chendo11/workfolder/Pfizer_project/PLFormer_output_models/model_cls_19'
    pretrained_model_path = r'/home/chendo11/workfolder/Pfizer_project/PLFormer_output_models/pretrained_plf_model'

    # load model
    config = TopTConfig.from_pretrained(original_model_path)
    model = TopTForImageClassification.from_pretrained(
        original_model_path, 
        # onfig=config
    )
    # print(config)
    original_state_dict = model.state_dict()
    new_state_dict = {}
    for key, value in original_state_dict.items():
        # print(key)
        new_key = key.replace('topt', 'plf')  # Replace 'topt' with 'plf'
        # print(new_key)
        new_state_dict[new_key] = value

    # creat the plf initial model
    # print(config)
    plf_config_dict = {}
    for k, v in config.to_dict().items():
        if k == 'architectures':
            v = 'PLFForImageClassification'
        elif k == 'model_type':
            v = 'plf'
        else:
            plf_config_dict[k] = v
    plf_config = PLFConfig()
    plf_config.update(plf_config_dict)
    plf_model = PLFForImageClassification.from_pretrained(pretrained_model_path, config=plf_config)
    # print(plf_model)

    # reset the parameters
    plf_model.load_state_dict(new_state_dict)

    # # save the model
    save_directory = r'/home/chendo11/workfolder/Pfizer_project/PLFormer_output_models/model_cls_19_pls' 
    plf_model.save_pretrained(save_directory)

    # # test the saved model
    config_test = PLFConfig.from_pretrained(save_directory)
    model = PLFForPreTraining.from_pretrained(save_directory, config=config)
    # print(config_test)

    return None


def load_and_save_batch_finetuned_model():
    save_folder = r'/home/chendo11/workfolder/Pfizer_project/output_models/finetuned_pls_model_by_2020_64_0.0008'
    original_folder = r'/home/chendo11/workfolder/TopTransformer/Output_dir/finetune_for_regression_v2020_010_20_general_rm_core/selected_global_para_64_0.00008_from_3w'

    pretrained_model_path = r'/home/chendo11/workfolder/Pfizer_project/output_models/pretrained_plf_model'

    for i in range(20):
        original_model_path = os.path.join(original_folder, f"model_cls_{i}")

        # load model
        config = TopTConfig.from_pretrained(original_model_path)
        model = TopTForImageClassification.from_pretrained(original_model_path)
        # print(config)
        original_state_dict = model.state_dict()
        new_state_dict = {}
        for key, value in original_state_dict.items():
            # print(key)
            new_key = key.replace('topt', 'plf')  # Replace 'topt' with 'plf'
            # print(new_key)
            new_state_dict[new_key] = value

        # creat the plf initial model
        # print(config)
        plf_config_dict = {}
        for k, v in config.to_dict().items():
            if k == 'architectures':
                v = 'PLFForImageClassification'
            elif k == 'model_type':
                v = 'plf'
            else:
                plf_config_dict[k] = v
        plf_config = PLFConfig()
        plf_config.update(plf_config_dict)
        plf_model = PLFForImageClassification.from_pretrained(pretrained_model_path, config=plf_config)
        # print(plf_model)

        # reset the parameters
        plf_model.load_state_dict(new_state_dict)

        # # save the model
        save_directory = os.path.join(save_folder, f"model_pls_cls_{i}")
        plf_model.save_pretrained(save_directory)

        # # # test the saved model
        # config_test = PLFConfig.from_pretrained(save_directory)
        # model = PLFForPreTraining.from_pretrained(save_directory, config=config)
        # print(config_test)
        print(i, 'converted')

    return None




def main():
    # get_predictions()
    # load_and_save_pretrained()
    # load_and_save_finetuned()
    load_and_save_batch_finetuned_model()
    return None


if __name__ == "__main__":
    main()
    print('End!')