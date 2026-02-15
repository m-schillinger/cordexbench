import xarray as xr
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from datetime import datetime
import argparse
import torch
import pdb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Ridge models for downscaling")
    parser.add_argument("--target_var", type=str, default="pr", help="Target variable (e.g., 'pr', 'tasmax')")
    parser.add_argument("--training_experiment", type=str, default="Emulator_hist_future", help="Training experiment name")
    parser.add_argument("--domain", type=str, default="ALPS", help="Domain name (e.g., 'ALPS', 'NZ', 'SA')")
    parser.add_argument("--root", type=str, default="/r/scratch/users/mschillinger/data/cordexbench/", help="Root data path")
    parser.add_argument("--mode", type=str, default="train", help="Mode: 'train' or 'predict'")
    parser.add_argument("--save_coefficients", action="store_true", default=True, help="Save Ridge coefficients")
    parser.add_argument("--save_scaler", action="store_true", default=True, help="Save scaler parameters")
    parser.add_argument('--past_future', action='store_true', help='Train model only on past')

    args = parser.parse_args()
    
    target_var = args.target_var
    root = args.root
    domain = args.domain
    training_experiment = args.training_experiment
    mode = args.mode
    save_coefficients = args.save_coefficients
    save_scaler = args.save_scaler
    
    if mode == "train":
        folder = "train"
    else:
        raise ValueError("In DownscalingDatasetNormed: mode not recognised")
            
    if training_experiment == 'ESD_pseudo_reality':
        period_training = '1961-1980'
    elif training_experiment == 'Emulator_hist_future':
        period_training = '1961-1980_2080-2099'
    else:
        raise ValueError('Provide a valid date')

    # Set the GCM
    if domain == 'ALPS':
        gcm_name = 'CNRM-CM5'
    elif domain == 'NZ' or domain == 'SA':
        gcm_name = 'ACCESS-CM2'
    DATA_PATH = root + f"/{domain}/{domain}_domain"
    hr_path = f'{DATA_PATH}/{folder}/{training_experiment}/target/pr_tasmax_{gcm_name}_{period_training}.nc'
    hr_ds = xr.open_dataset(hr_path)

    lr_path = f'{DATA_PATH}/{folder}/{training_experiment}/predictors/{gcm_name}_{period_training}.nc'
    lr_ds = xr.open_dataset(lr_path).astype("float32")
    lr_ds

    # Fit a separate Ridge model for multiple HR locations and save predictions (past + future)

    # Select a subset of locations to reduce compute (adjust step_y/step_x as needed)
    step_y, step_x = 1, 1
    # Handle HR grid dims that may be named 'y'/'x' or 'lat'/'lon'
    hr_y_name = 'y' if 'y' in hr_ds.dims else ('lat' if 'lat' in hr_ds.dims else None)
    hr_x_name = 'x' if 'x' in hr_ds.dims else ('lon' if 'lon' in hr_ds.dims else None)
    if hr_y_name is None or hr_x_name is None:
        raise ValueError(f"HR dataset has unexpected dims: {list(hr_ds.dims)}; expected 'y'/'x' or 'lat'/'lon'.")
    y_indices = np.arange(0, hr_ds.sizes[hr_y_name], step_y)
    x_indices = np.arange(0, hr_ds.sizes[hr_x_name], step_x)

    # Prepare output arrays
    preds = xr.DataArray(
        np.empty((lr_ds.sizes['time'], len(y_indices), len(x_indices)), dtype=np.float32),
        coords={'time': lr_ds.time, 'y': hr_ds[hr_y_name].values[y_indices], 'x': hr_ds[hr_x_name].values[x_indices]},
        dims=['time', 'y', 'x'],
        name=f'{target_var}_ridge_pred',
        attrs={'units': hr_ds[target_var].attrs.get('units', '')}
    )
    r2_past = xr.DataArray(
        np.empty((len(y_indices), len(x_indices)), dtype=np.float32),
        coords={'y': hr_ds[hr_y_name].values[y_indices], 'x': hr_ds[hr_x_name].values[x_indices]},
        dims=['y', 'x'],
        name=f'{target_var}_ridge_r2_past'
    )
    r2_future = xr.DataArray(
        np.empty((len(y_indices), len(x_indices)), dtype=np.float32),
        coords={'y': hr_ds[hr_y_name].values[y_indices], 'x': hr_ds[hr_x_name].values[x_indices]},
        dims=['y', 'x'],
        name=f'{target_var}_ridge_r2_future'
    )

    # Use the alpha from the existing pipeline if available, else fallback
    alpha_val = 1000.0
    if 'time_bnds' in lr_ds.data_vars:
        lr_ds = lr_ds.drop_vars('time_bnds')
    sorted_features = sorted(list(lr_ds.data_vars))
    features_da = lr_ds[sorted_features].to_array()  # dims: variable
    
    features_da = features_da.transpose('time', 'lat', 'lon', 'variable')
    X = features_da.values.reshape(features_da.sizes['time'], -1)

    if args.past_future:
        maxpoint = len(X) // 2
    else:
        maxpoint = len(X)


    # Train per-location on past data, predict past + future
    def _fit_predict_for_location(yi_i, xi_i, yi_idx, xi_idx):
        
        y_loc_da = hr_ds[target_var].isel({hr_y_name: yi_idx, hr_x_name: xi_idx}).sel(time=lr_ds.time)
        y_loc = y_loc_da.values
        
        if target_var == 'pr':
            y_loc = np.sqrt(y_loc)
        elif target_var == 'tasmax':
            y_loc = y_loc

        X_train = X[:maxpoint]
        y_train = y_loc[:maxpoint]

        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha_val))
        model.fit(X_train, y_train)

        pred_past = model.predict(X[:maxpoint]).astype(np.float32)
        if args.past_future:
            pred_future = model.predict(X[maxpoint:]).astype(np.float32)
        else:
            pred_future = np.array([], dtype=np.float32)
            
        r2_p = model.score(X_train, y_train)
        
        if args.past_future:
            r2_f = model.score(X[maxpoint:], y_loc[maxpoint:])
        else:
            r2_f = np.nan

        scaler = model.named_steps['standardscaler']
        ridge = model.named_steps['ridge']
        scale = scaler.scale_
        mean = scaler.mean_
        coef = ridge.coef_
        safe_scale = np.where(scale == 0, 1.0, scale)
        eff_coef = coef / safe_scale
        # Effective intercept for raw features
        eff_intercept = ridge.intercept_ - np.dot(mean[scale != 0], (coef[scale != 0] / scale[scale != 0]))
        # Return per-location scaler params so they can be saved
        return (
            yi_i,
            xi_i,
            pred_past,
            pred_future,
            r2_p,
            r2_f,
            eff_coef.astype(np.float32),
            np.float32(eff_intercept),
            mean.astype(np.float32),
            scale.astype(np.float32),
        )

    results = Parallel(n_jobs=50, backend="loky")(
        delayed(_fit_predict_for_location)(yi_i, xi_i, yi, xi)
        for yi_i, yi in enumerate(y_indices)
        for xi_i, xi in enumerate(x_indices)
    )

    # Prepare coefficient and (optional) per-location scaler containers
    if save_coefficients:
        coefs = xr.DataArray(
            np.empty((len(y_indices), len(x_indices), features_da.sizes['lat'], features_da.sizes['lon'], features_da.sizes['variable']), dtype=np.float32),
            coords={
                'y': hr_ds[hr_y_name].values[y_indices],
                'x': hr_ds[hr_x_name].values[x_indices],
                'lat': lr_ds.lat.values,
                'lon': lr_ds.lon.values,
                'variable': sorted_features
            },
            dims=['y', 'x', 'lat', 'lon', 'variable'],
            name=f'{target_var}_ridge_coefficients'
        )
    if save_scaler:
        scaler_mean = xr.DataArray(
            np.empty((len(y_indices), len(x_indices), features_da.sizes['lat'], features_da.sizes['lon'], features_da.sizes['variable']), dtype=np.float32),
            coords={
                'y': hr_ds[hr_y_name].values[y_indices],
                'x': hr_ds[hr_x_name].values[x_indices],
                'lat': lr_ds.lat.values,
                'lon': lr_ds.lon.values,
                'variable': sorted_features
            },
            dims=['y', 'x', 'lat', 'lon', 'variable'],
            name=f'{target_var}_ridge_scaler_mean'
        )
        scaler_scale = xr.DataArray(
            np.empty((len(y_indices), len(x_indices), features_da.sizes['lat'], features_da.sizes['lon'], features_da.sizes['variable']), dtype=np.float32),
            coords={
                'y': hr_ds[hr_y_name].values[y_indices],
                'x': hr_ds[hr_x_name].values[x_indices],
                'lat': lr_ds.lat.values,
                'lon': lr_ds.lon.values,
                'variable': sorted_features
            },
            dims=['y', 'x', 'lat', 'lon', 'variable'],
            name=f'{target_var}_ridge_scaler_scale'
        )

    # Intercepts container (effective, for raw features)
    intercepts = xr.DataArray(
        np.empty((len(y_indices), len(x_indices)), dtype=np.float32),
        coords={'y': hr_ds[hr_y_name].values[y_indices], 'x': hr_ds[hr_x_name].values[x_indices]},
        dims=['y', 'x'],
        name=f'{target_var}_ridge_intercepts'
    )

    for yi_i, xi_i, pred_past, pred_future, r2_p, r2_f, loc_coef, loc_intercept, loc_mean, loc_scale in results:
        if target_var == 'pr':
            pred_past = np.maximum(pred_past, 0)  # Ensure non-negativity before squaring
            pred_future = np.maximum(pred_future, 0)
            pred_past = pred_past ** 2
            pred_future = pred_future ** 2
        preds.values[:maxpoint, yi_i, xi_i] = pred_past
        preds.values[maxpoint:, yi_i, xi_i] = pred_future
        r2_past.values[yi_i, xi_i] = r2_p
        r2_future.values[yi_i, xi_i] = r2_f
        if save_coefficients:
            loc_coef_reshaped = loc_coef.reshape(features_da.sizes['lat'], features_da.sizes['lon'], features_da.sizes['variable'])
            coefs.values[yi_i, xi_i, :, :, :] = loc_coef_reshaped
        intercepts.values[yi_i, xi_i] = loc_intercept
        if save_scaler:
            scaler_mean.values[yi_i, xi_i, :, :, :] = loc_mean.reshape(features_da.sizes['lat'], features_da.sizes['lon'], features_da.sizes['variable'])
            scaler_scale.values[yi_i, xi_i, :, :, :] = loc_scale.reshape(features_da.sizes['lat'], features_da.sizes['lon'], features_da.sizes['variable'])

    # Save results to NetCDF with versioned fallback on permission issues
    def save_with_version_fallback(data_obj, base_path, encoding=None):
        out_dir = os.path.dirname(base_path)
        os.makedirs(out_dir, exist_ok=True)

        # If directory itself isn't writable, surface a clear error
        if not os.access(out_dir, os.W_OK):
            raise PermissionError(f"Output directory not writable: {out_dir}")

        try:
            data_obj.to_netcdf(base_path, encoding=encoding)
            return base_path
        except PermissionError:
            root, ext = os.path.splitext(base_path)
            ts = datetime.now().strftime('%Y%m%d-%H%M%S')
            alt_path = f"{root}_{ts}{ext}"
            data_obj.to_netcdf(alt_path, encoding=encoding)
            return alt_path

    if args.past_future:
        name = "trained-on-past/"
    else:
        name = ""
    
    # save predictions separately
    output_path = f'{DATA_PATH}/{folder}/{training_experiment}/linear_models/{name}ridge_{target_var}_{gcm_name}_{period_training}_alpha-{alpha_val}_preds.nc'
    saved_path = save_with_version_fallback(preds, output_path)
    print(f"Saved predictions to: {saved_path}")
    

    name_str = "" 
        
    data_type = target_var

    file_base = f"{training_experiment}_{data_type}_{gcm_name}_{period_training}{name_str}_alpha-{alpha_val}"
    torch.save({"pred": torch.tensor(
        np.flip(preds.values, [1]).copy())},
        os.path.join(root, domain, "norm_stats", f"hr_norm_stats_linear-pred_all_{file_base}.pt"))
    
    torch.save({"pred": torch.tensor(
        np.sqrt(np.flip(preds.values, [1]).copy()))},
        os.path.join(root, domain, "norm_stats", f"hr_norm_stats_linear-pred_all_{file_base}_sqrt.pt"))

    if save_coefficients:
        coef_path = f'{DATA_PATH}/{folder}/{training_experiment}/linear_models/{name}ridge_{target_var}_{gcm_name}_{period_training}_alpha-{alpha_val}_coefs.nc'
        saved_coef_path = save_with_version_fallback(coefs, coef_path)
        print(f"Saved coefficients to: {saved_coef_path}")
        inter_path = f'{DATA_PATH}/{folder}/{training_experiment}/linear_models/{name}ridge_{target_var}_{gcm_name}_{period_training}_alpha-{alpha_val}_intercepts.nc'
        saved_inter_path = save_with_version_fallback(intercepts, inter_path)
        print(f"Saved effective intercepts to: {saved_inter_path}")

    if save_scaler:
        scaler_ds = xr.Dataset({'scaler_mean': scaler_mean, 'scaler_scale': scaler_scale})
        scaler_path = f'{DATA_PATH}/{folder}/{training_experiment}/linear_models/{name}ridge_{target_var}_{gcm_name}_{period_training}_alpha-{alpha_val}_perloc_scaler.nc'
        saved_scaler_path = save_with_version_fallback(scaler_ds, scaler_path)
        print(f"Saved per-location scaler params to: {saved_scaler_path}")