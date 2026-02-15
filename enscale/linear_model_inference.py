import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Train Ridge models for downscaling")
    parser.add_argument("--target_var", type=str, default="pr", help="Target variable (e.g., 'pr', 'tasmax')")
    parser.add_argument("--training_experiment", type=str, default="Emulator_hist_future", help="Training experiment name")
    parser.add_argument("--domain", type=str, default="ALPS", help="Domain name (e.g., 'ALPS', 'NZ', 'SA')")
    parser.add_argument("--root", type=str, default="/r/scratch/users/mschillinger/data/cordexbench/", help="Root data path")

    args = parser.parse_args()
    
    target_var = args.target_var
    root = args.root
    domain = args.domain
    training_experiment = args.training_experiment
    
    # Configuration
    if training_experiment == 'ESD_pseudo_reality':
        period_training = '1961-1980'
    elif training_experiment == 'Emulator_hist_future':
        period_training = '1961-1980_2080-2099'
    else:
        raise ValueError('Provide a valid date')    
    folder = 'train'


    # Choose GCM by domain (match training)
    if domain == 'ALPS':
        gcm_name = 'CNRM-CM5'
        alpha_val = 100.0
    elif domain in ('NZ', 'SA'):
        gcm_name = 'ACCESS-CM2'
        alpha_val = 1000.0
    else:
        raise ValueError('Unknown domain')

    DATA_PATH = f"{root}/{domain}/{domain}_domain"
    predictors_path = f"{DATA_PATH}/{folder}/{training_experiment}/predictors/{gcm_name}_{period_training}.nc"
    coef_path = f"{DATA_PATH}/{folder}/{training_experiment}/linear_models/ridge_{target_var}_{gcm_name}_{period_training}_alpha-{alpha_val}_coefs.nc"
    inter_path = f"{DATA_PATH}/{folder}/{training_experiment}/linear_models/ridge_{target_var}_{gcm_name}_{period_training}_alpha-{alpha_val}_intercepts.nc"
    scaler_path = f"{DATA_PATH}/{folder}/{training_experiment}/linear_models/ridge_{target_var}_{gcm_name}_{period_training}_alpha-{alpha_val}_perloc_scaler.nc"  # optional

    
    # to do: orientation? flipping?
    test_periods = ["historical", "mid_century", "end_century"]
    frameworks = ["perfect", "imperfect"]
    if args.domain == "ALPS":
        gcms = ["MPI-ESM-LR", "CNRM-CM5"]
    elif args.domain == "NZ":
        gcms = ["ACCESS-CM2", "EC-Earth3"]
    elif args.domain == "SA":
        gcms = ["ACCESS-CM2", "NorESM2-MM"]
    
    for mode in ["inference"]:
        print(f"--- Processing Mode: {mode} ---")
        
        # 1. Determine iterations
        # If mode is train, we run once. If test, we loop through params.
        configs = [
            {"period": p, "framework": f, "gcm": g}
            for p in test_periods for f in frameworks for g in gcms
        ]

        for test_params in configs:
            if test_params["period"] == "historical":
                period_inference = "1981-2000"
            elif test_params["period"] == "mid_century":
                period_inference = "2041-2060"
            elif test_params["period"] == "end_century":
                period_inference = "2080-2099"  
            
            print(f"Processing {test_params['period']} - {test_params['framework']} - {test_params['gcm']}")
                
            lr_path = f'{DATA_PATH}/test/{test_params["period"]}/predictors/{test_params["framework"]}/{test_params["gcm"]}_{period_inference}.nc'

            # Load predictors and model artifacts
            lr_ds = xr.open_dataset(lr_path)
            print("reading coefs from", coef_path)
            coefs = xr.open_dataarray(coef_path)
            intercepts = xr.open_dataarray(inter_path)
            # scaler_ds = xr.open_dataset(scaler_path)  # not required when using effective weights

            # Align predictor order with saved coefficients
            sorted_vars = lr_ds.data_vars
            # remove time bounds
            lr_ds = lr_ds[[v for v in sorted_vars if v != "time_bnds"]]
            lr_ds.astype('float32')
            features_da = lr_ds.to_array()  # dims: variable, time?, lat, lon
            features_da = features_da.transpose('time', 'lat', 'lon', 'variable')
            features_da = features_da.sel(variable=coefs['variable'])
            X = features_da.values.astype(np.float32)             # (time, lat, lon, variable)
            coef_vals = coefs.values.astype(np.float32)           # (y, x, lat, lon, variable)

            # Contract predictors with coefficients for all locations
            preds = np.einsum('tijk,xyijk->txy', X, coef_vals).astype(np.float32)  # (time, y, x)
            # Add effective intercepts per (y,x)
            preds += intercepts.values[None, :, :].astype(np.float32)
            
            os.makedirs(os.path.join(
                        "/r/scratch/groups/nm/downscaling/samples_cordexbench/", training_experiment, domain, "no-orog", target_var, "linear_pred"), exist_ok=True)
            ns_path = os.path.join(
                        "/r/scratch/groups/nm/downscaling/samples_cordexbench/", training_experiment, domain, "no-orog", target_var, "linear_pred",
                        f"linear-pred_{test_params['period']}_{test_params['framework']}_{test_params['gcm']}_alpha-{alpha_val}.pt")
            
            torch.save({"pred": torch.from_numpy(
                                np.flip(preds, [1]).copy() 
                                )}, ns_path)
                    