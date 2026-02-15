import os
from pathlib import Path
import pdb
from typing import List, Optional, Tuple

import numpy as np
import xarray as xr
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.linear_model import LinearRegression


def _training_gcm_for_domain(domain: str) -> str:
	"""Return the GCM name used for training predictors based on domain.

	Matches the project's convention:
	- ALPS -> CNRM-CM5
	- NZ, SA -> ACCESS-CM2
	"""
	if domain == "ALPS":
		return "CNRM-CM5"
	elif domain in ("NZ", "SA"):
		return "ACCESS-CM2"
	raise ValueError(f"Unsupported domain: {domain}")


def _compute_forced_signal_lowess(series: np.ndarray, t_days: np.ndarray, window_days: int) -> np.ndarray:
    """Compute forced signal via LOWESS smoothing of the spatial-mean t_500.

    Parameters
    - series: 1D array of t_500 mean over space across concatenated time
    - t_days: 1D array of days since start corresponding to series
    - window_days: window length in days used to derive frac for LOWESS

    NaNs are excluded from the LOWESS fit; the smooth is interpolated back to all times.
    """
    # Identify valid (non-NaN) points
    valid_mask = ~np.isnan(series)
    if np.sum(valid_mask) < 5:
        # Too few valid points; fallback to linear fit on valid data
        X_valid = t_days[valid_mask].reshape(-1, 1)
        s_valid = series[valid_mask]
        lr = LinearRegression()
        lr.fit(X_valid, s_valid)
        # Predict on all times
        X_all = t_days.reshape(-1, 1)
        return lr.predict(X_all)

    # Fit LOWESS on valid data only
    t_valid = t_days[valid_mask]
    s_valid = series[valid_mask]

    frac = max(0.01, min(0.99, window_days / float(len(s_valid))))
    pred_valid = lowess(s_valid, t_valid, frac=frac, it=0, return_sorted=False)

    # Interpolate back to all time points
    pred_all = np.interp(t_days, t_valid, pred_valid, left=pred_valid[0], right=pred_valid[-1])
    return pred_all


def _fit_linear_on_forced(x_anom: xr.DataArray, y: xr.DataArray) -> dict:
	"""Fit linear regression of y on the forced-signal anomaly x_anom.

	NaN values are excluded from the fit. The fitted model is applied to all times,
	preserving NaNs in the output where they existed in the input.

	Returns dict with slope, intercept, pred, and detrended as DataArrays.
	"""
	# Ensure alignment
	y = y.sel(time=x_anom.time)
	X = x_anom.values.reshape(-1, 1)  # (T, 1)
	Y = y.values.reshape(X.shape[0], -1)  # (T, N)

	# Identify valid (non-NaN) points across all spatial dims
	# We want to exclude any time step that has NaN in X or any spatial location in Y
	valid_time_mask = ~np.isnan(X[:, 0])  # Valid where forced signal is not NaN
	if Y.ndim > 1:
		# Also exclude time steps where any spatial location has NaN
		valid_time_mask &= ~np.any(np.isnan(Y), axis=tuple(range(1, Y.ndim)))
	else:
		valid_time_mask &= ~np.isnan(Y)

	if np.sum(valid_time_mask) < 2:
		# Not enough valid points to fit; return NaNs
		print("[WARN] Insufficient valid data points for regression (need >= 2).")
		linear_pred = np.full_like(Y, np.nan)
		detrended_vals = np.full_like(Y, np.nan)
		slope = np.full(Y.shape[1:], np.nan)
		intercept = np.full(Y.shape[1:], np.nan)
	else:
		X_valid = X[valid_time_mask]
		Y_valid = Y[valid_time_mask]

		model = LinearRegression()
		model.fit(X_valid, Y_valid)

		slope = model.coef_[:, 0] if Y_valid.ndim > 1 else model.coef_[0]  # (N,) or scalar
		intercept = model.intercept_  # (N,) or scalar
		linear_pred = np.full_like(Y, np.nan)
		detrended_vals = np.full_like(Y, np.nan)

		# Apply model to all times
		linear_pred_all = model.predict(X)  # (T, N)
		detrended_all = Y - linear_pred_all + intercept  # (T, N)

		# Keep NaNs where they were
		linear_pred[valid_time_mask] = linear_pred_all[valid_time_mask]
		detrended_vals[valid_time_mask] = detrended_all[valid_time_mask]

	# Reshape back to original y shape
	detrended_vals = detrended_vals.reshape(y.shape)
	linear_pred = linear_pred.reshape(y.shape)
	slope = slope.reshape(y.shape[1:]) if isinstance(slope, np.ndarray) and slope.ndim > 0 else slope
	intercept = intercept.reshape(y.shape[1:]) if isinstance(intercept, np.ndarray) and intercept.ndim > 0 else intercept

	return {
		"slope": xr.DataArray(slope, coords={k: y.coords[k] for k in y.dims if k != "time"}, dims=[d for d in y.dims if d != "time"]),
		"intercept": xr.DataArray(intercept, coords={k: y.coords[k] for k in y.dims if k != "time"}, dims=[d for d in y.dims if d != "time"]),
		"pred": xr.DataArray(linear_pred, coords=y.coords, dims=y.dims),
		"detrended": xr.DataArray(detrended_vals, coords=y.coords, dims=y.dims),
	}


def detrend_single_inference_file(
	root: str,
	domain: str,
	inf_path: Path,
	window_years: int = 20,
	training_experiment: str = "ESD_pseudo_reality",
) -> Optional[Path]:
    """Detrend predictors for a single inference file by concatenating with training predictors.

    Saving alongside the inference file with suffix `_climate-change-lin-detrended.nc`.

    Returns the output path if created, else None.
    """
    gcm_train = _training_gcm_for_domain(domain)
    period_training = "1961-1980"

    data_path = os.path.join(root, domain, f"{domain}_domain")
    train_path = Path(data_path) / "train" / training_experiment / "predictors" / f"{gcm_train}_{period_training}.nc"

    if not train_path.exists():
        print(f"[WARN] Training predictors not found: {train_path}")
        return None

    try:
        ds_train = xr.open_dataset(train_path)
        ds_inf = xr.open_dataset(inf_path)
    except Exception as e:
        print(f"[ERROR] Failed opening datasets: {e}\n  train: {train_path}\n  inf:   {inf_path}")
        return None

    # Intersect variables to be safe
    common_vars = [v for v in ds_train.data_vars if v in ds_inf.data_vars and v != "time_bnds"]
    if len(common_vars) == 0:
        print(f"[WARN] No common predictor variables between training and inference for {inf_path}")
        return None
    common_vars = sorted(common_vars)  # Sort for consistent order

    # Concatenate along time
    # Ensure identical spatial dims order
    # We'll select only common vars to avoid alignment issues.
    ds_train_sel = ds_train[common_vars]
    ds_inf_sel = ds_inf[common_vars]

    # Basic sanity on lat/lon shape
    for dim in ("lat", "lon"):
        if (dim not in ds_train_sel.dims) or (dim not in ds_inf_sel.dims):
            print(f"[WARN] Missing spatial dim '{dim}' in one of the datasets for {inf_path}")
            return None
        if ds_train_sel.sizes[dim] != ds_inf_sel.sizes[dim]:
            print(f"[WARN] Spatial dimension mismatch for {inf_path}: {dim} train={ds_train_sel.sizes[dim]} inf={ds_inf_sel.sizes[dim]}")
            return None

    combined = xr.concat([ds_train_sel, ds_inf_sel], dim="time")


    # Compute spatial mean across lat/lon
    lr_mean = combined.to_array().mean(dim=("lon", "lat"))

    # Use t_500 as proxy for forced signal (matches notebook approach)
    if "t_500" not in lr_mean["variable"].values:
        print(f"[WARN] 't_500' not found in predictors for {inf_path}. Skipping.")
        return None

    t500_series = lr_mean.sel(variable="t_500").values
    t_values = lr_mean.time.values.astype("datetime64[D]")
    t_days = (t_values - t_values[0]).astype(float)
    window_days = int(window_years * 365)

    if len(t500_series.shape) > 1:
        pdb.set_trace()

    forced = _compute_forced_signal_lowess(t500_series, t_days, window_days)
    # Convert to anomaly relative to the start
    forced_anom = forced - forced[0]

    x = xr.DataArray(forced_anom, dims=["time"], coords={"time": combined.time})

    # Fit and detrend each predictor variable
    results = {}
    for var in common_vars:
        results[var] = _fit_linear_on_forced(x, combined[var])

    # Build detrended Dataset and slice to inference period only
    detrended_combined = xr.Dataset({var: results[var]["detrended"] for var in common_vars})
    detrended_inf = detrended_combined.sel(time=ds_inf_sel.time)

    # Save alongside the inference file
    out_path = inf_path.with_name(inf_path.stem + "_climate-change-lin-detrended.nc")
    try:
        detrended_inf.to_netcdf(out_path)
        print(f"[OK] Saved detrended predictors -> {out_path}")
        return out_path
    except Exception as e:
        print(f"[ERROR] Failed saving detrended NetCDF: {e}\n  target: {out_path}")
        return None


def find_inference_files(root: str, domain: str) -> List[Path]:
	"""Find all inference predictor files matching the expected folder layout."""
	base = Path(root) / domain / f"{domain}_domain" / "test"
	# Pattern: test/{period}/predictors/{framework}/{gcm}_{period_inference}.nc
	return list(base.glob("*/predictors/*/*.nc"))


def main():
	import argparse

	parser = argparse.ArgumentParser(description="Detrend inference predictors by regressing on smoothed t_500 trend, using training concatenation.")
	parser.add_argument("--root", type=str, default="/r/scratch/users/mschillinger/data/cordexbench/", help="Root data directory")
	parser.add_argument("--domain", type=str, default="ALPS", choices=["ALPS", "NZ", "SA"], help="Domain to process")
	parser.add_argument("--window-years", type=int, default=20, help="LOWESS window in years for forced signal")
	parser.add_argument("--overwrite", action="store_true", help="Overwrite existing detrended files")
	parser.add_argument("--dry-run", action="store_true", help="List files to be processed without writing outputs")

	args = parser.parse_args()

	inf_files = find_inference_files(args.root, args.domain)
	if not inf_files:
		print(f"[INFO] No inference predictor files found under {args.root}{args.domain}/{args.domain}_domain/test")
		return

	print(f"[INFO] Found {len(inf_files)} inference files. Processing...")

	for inf_path in sorted(inf_files):
		out_path = inf_path.with_name(inf_path.stem + "_climate-change-lin-detrended.nc")
		if out_path.exists() and not args.overwrite:
			print(f"[SKIP] Exists (use --overwrite to replace): {out_path}")
			continue
		if args.dry_run:
			print(f"[DRY] Would process: {inf_path}")
			continue
		detrend_single_inference_file(
			root=args.root,
			domain=args.domain,
			inf_path=inf_path,
			window_years=args.window_years,
			training_experiment="ESD_pseudo_reality",
		)


if __name__ == "__main__":
	main()

