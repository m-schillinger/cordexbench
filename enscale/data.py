import os

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torch.utils.data.dataset import ConcatDataset
from torch.distributions.normal import Normal
import os
import numpy as np
import re
import xarray as xr
from utils import *
import scipy

# -------------- DATASET CLASS ---------------------------------------

def get_normed_data(lr_ds, data_type, norm_input, sqrt_transform_in, root, domain="ALPS", training_experiment = 'Emulator_hist_future', 
                   gcm_name="CNRM-CM5", period_training="1961-1980_2080-2099", period_norm="1961-1980"):
    
    lr_data = torch.flip(torch.from_numpy(lr_ds[data_type].data), [1])
    lr_data = lr_data.float()
    # data = torch.from_numpy(lr_ds[data_type].data)
    # bring lr_data to the right units
    lr_data_norm = normalise(lr_data, mode = "lr", data_type = data_type, sqrt_transform = sqrt_transform_in, norm_method = norm_input, root=root,
                             logit=False, normal=False,
                            domain=domain,
                            training_experiment=training_experiment,
                            gcm_name=gcm_name,
                            period_training=period_training,
                            period=period_norm)
        
    return lr_data_norm

class DownscalingDatasetTwoStepNormed(Dataset):
    """
    Downscaling data loading class for two-step downscaling (GCM to coarse RCM to fine RCM).
    Loads data from local .nc files, normalises it and returns pairs of low-res and high-res data.
    1. Loads low-res GCM data and normalises it
    2. Loads high-res RCM data and normalises it
    3. Coarsens high-res RCM data to coarse RCM resolution (by average pooling)
    4. Returns (low-res GCM data, coarsened high-res RCM data) as input and high-res RCM data as output
    5. Optionally, appends time information (day of year) and one-hot encoding of GCM and RCM to the input
    
    Returns either:
    - (x, z, y) where x is the low-res GCM data + time + one-hot, z is the coarsened high-res RCM data, y is the high-res RCM data
    - (x, z, y, x_next, z_next, y_next) if return_timepair=True, where x_next, z_next, y_next are the data for the next time step (used for temporal consistency training)
    
    Parameters
    ----------
    - root: path to files
    - domain: "ALPS" or "NZ"
    - training_experiment: 'ESD_pseudo_reality' or 'Emulator_hist_future'
    - data_types: list of variables to load for high-res RCM data (e.g. ["pr", "tas"])
    - data_types_lr: list of variables to load for low-res GCM data (e.g. ["pr", "tas", "sfcWind", "rsds", "psl"])
    - mode: "train", "test_interpolation" or "test_extrapolation" (used only to find correct folder for data loading)
    - norm_input: normalisation method for low-res GCM data
    - norm_output: normalisation method for high-res RCM data
    - sqrt_transform_in: if True, apply square-root transform to low-res GCM data
    - sqrt_transform_out: if True, apply square-root transform to high-res RCM data
    - kernel_size: kernel size for average pooling to coarsen high-res RCM data
    - kernel_size_hr: kernel size for additional coarsening of the target high-res RCM data (used in intermediate steps of enscale's super-resolution model)
    - return_timepair: if True, return also the data for the next time step (used for temporal consistency training)
    - clip_quantile: if not None, clip high-res RCM data to the given quantile
    - logit: if True, apply logit transform to high-res RCM data (only useful if norm_output is "uniform" or "uniform_per_model")
    - normal: if True, apply normal transform to high-res RCM data (only useful norm_output is "uniform" or "uniform_per_model")
    - include_year: if True, include year as input feature (in addition to day of year)
    - only_winter: if True, load only data from December, January, February
    - stride_lr: stride for average pooling of low-res GCM data (default: kernel_size)
    - padding_lr: padding for average pooling of low-res GCM data (default: 0)
    - filter_outliers: if True, load data from outlier-filtered files (only for high-res RCM data; currently filtering is done separately and saved on disk, only for "pr" and "rsds")
    - precip_zeros: how to treat zeros in precipitation data when using norm_output="uniform" or "uniform_per_model"; also done separately and saved on disk (options: "random", "random_correlated", "constant")
    """
    def __init__(self, 
                 root="/r/scratch/users/mschillinger/data/cordexbench/",
                 domain="ALPS",
                 training_experiment = 'Emulator_hist_future',
                 data_types=["tas", "pr", "sfcWind", "rsds"],
                 data_types_lr = None,
                 mode = "train", # or "inference"
                 norm_input=None,
                 norm_output=None,
                 sqrt_transform_in=True,
                 sqrt_transform_out=True,
                 # kernel_size=[16],
                 kernel_size=16,
                 kernel_size_hr=1,
                 return_timepair=False,
                 clip_quantile=None, 
                 logit=False,
                 normal=False,
                 include_year=False,
                 stride_lr = None,
                 padding_lr = None,
                 orog_file="Static_fields.nc",
                 period_norm="1961-1980",
                 remove_climate_change_trend=False,
                 test_params={"period": "historical",
                              "gcm": "CNRM-CM5",
                              "framework": "perfect"},
                 alpha_subtract_linear=100
                 ):
        """
        Notes
        - In general, one has to pay attention with the orientation of the data: If one converts the data from a .nc file to a tensor, sometimes the first rows will correspond to South, sometimes to North.
        Therefore, we sometimes have to flip the data - we agree on "first rows = north.
        """
        self.return_timepair = return_timepair
        self.mode = mode
        # {domain}_domain
        DATA_PATH = root + f"{domain}/{domain}_domain"
        
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
        if mode == "train":
            folder = "train"
        else:
            if test_params["period"] == "historical":
                period_inference = "1981-2000"
            elif test_params["period"] == "mid_century":
                period_inference = "2041-2060"
            elif test_params["period"] == "end_century":
                period_inference = "2080-2099"  
                
        hr_tensors = []
        hr_coarsened_tensors = []
        lr_tensors = []
        # loop through variables and concatenate along first dimension
        # shape in the end should be [num_variables, n_channels, image_size, image_size]
        # n_channels is 1 for now, but 2 later after fft
        
        if norm_output == "uniform_per_model" or norm_output == "uniform": 
            raise NotImplementedError("Please use norm_output unequal to uniform or uniform_per_model, as these have been removed for now.")     
                
        else:
            if mode == "train":
                # example: pr_day_EUR-11_ALADIN63_MPI-ESM-LR_r1i1p1_rcp85_ALPS_cordexgrid_train-period.nc
                hr_path = f'{DATA_PATH}/{folder}/{training_experiment}/target/pr_tasmax_{gcm_name}_{period_training}.nc'
                # hr_path = os.path.join (root, folder, data_type + "_day_EUR-11_" + rcm + "_" + gcm + "_" + variant + "_rcp85_ALPS_cordexgrid" +  file_suffix + ".nc")
                # flipping the data to have correct north-south orientation
                print(hr_path)
                hr_ds = xr.open_dataset(hr_path)
                
                for data_type in data_types:
                    
                    hr_data = torch.flip(torch.from_numpy(hr_ds[data_type].data), [1])
                    hr_data = hr_data.float()
                    if data_type == "pr" and clip_quantile is not None:
                        # hr_data = torch.clamp(hr_data, torch.quantile(hr_data, 1 - clip_quantile).item(), torch.quantile(hr_data, clip_quantile).item())
                        # changed to batchwise clipping to shorten preprocessing time
                        hr_data = batchwise_quantile_clipping(hr_data, q = clip_quantile, batch_size = 365)
                        
                    hr_data_norm = normalise(hr_data, mode = "hr", data_type = data_type, 
                                            sqrt_transform = sqrt_transform_out, norm_method = norm_output, root=root,
                                            logit=logit, normal=normal,
                                            domain=domain, training_experiment=training_experiment, 
                                            gcm_name=gcm_name,
                                            period_training=period_training,
                                            alpha=alpha_subtract_linear,)
                    # if numpy array, convert to torch tensor again
                    if isinstance(hr_data_norm, np.ndarray):
                        hr_data_norm = torch.from_numpy(hr_data_norm)
                    
                if kernel_size_hr > 1:
                    hr_coarsened = torch.nn.functional.avg_pool2d(
                        hr_data_norm.view(hr_data_norm.shape[0], 128, 128).unsqueeze(1), kernel_size=kernel_size_hr, stride=kernel_size_hr).view(hr_data_norm.shape[0], 1, -1)
                    hr_tensors.append(hr_coarsened)
                else:
                    hr_tensors.append(hr_data_norm.unsqueeze(1))
                
                if kernel_size > 1:
                    k = kernel_size
                    if stride_lr is None:
                        stride_lr = k
                    if padding_lr is None:
                        padding_lr = 0
                    hr_coarsened = torch.nn.functional.avg_pool2d(
                        # hr_data_norm.view(hr_data_norm.shape[0], 128, 128).unsqueeze(1), kernel_size=k, stride=k//2, padding=k//2).view(hr_data_norm.shape[0], 1, -1)
                        # hr_data_norm.view(hr_data_norm.shape[0], 128, 128).unsqueeze(1), kernel_size=k, stride=k//2, padding=0).view(hr_data_norm.shape[0], 1, -1)
                        hr_data_norm.view(hr_data_norm.shape[0], 128, 128).unsqueeze(1), kernel_size=k, stride=stride_lr, padding=padding_lr).view(hr_data_norm.shape[0], 1, -1)
                    hr_coarsened_tensors.append(hr_coarsened)
                else:
                    hr_coarsened_tensors.append(hr_data_norm.unsqueeze(1))   

                hr_data_allvars = torch.concat(hr_tensors, dim = 1) # shape (n_timesteps, n_vars, spatial_dim), where spatial_dim = 128*128 or 128*128 + value_dim
                hr_data_coarsened_allvars = torch.concat(hr_coarsened_tensors, dim = 1)
                
        if mode == "train":
            if remove_climate_change_trend:
                # lr_path = f'{root}/{domain}/norm_stats/{training_experiment}_{gcm_name}_{period_training}_no-climate-change-trend.nc'
                lr_path = f'{root}/{domain}/norm_stats/{training_experiment}_{gcm_name}_{period_training}_climate-change-lin-detrended.nc'
                print("Loading detrending data from:", lr_path)
            else:
                lr_path = f'{DATA_PATH}/{folder}/{training_experiment}/predictors/{gcm_name}_{period_training}.nc'
            
            lr_ds = xr.open_dataset(lr_path)
            # lr_ds = lr_ds.astype("float32")
            if len(lr_ds.time) != len(hr_ds.time):
                print("Length mismatch. Should investigate further!")
        else:
            if remove_climate_change_trend:
                lr_path = f'{DATA_PATH}/test/{test_params["period"]}/predictors/{test_params["framework"]}/{test_params["gcm"]}_{period_inference}_climate-change-lin-detrended.nc'
                lr_ds = xr.open_dataset(lr_path)    
            else:    
                # DATA_PATH/test/test_params["period"/predictors/test_params["framework"]/ 
                # and file name gcm_name_period
                lr_path = f'{DATA_PATH}/test/{test_params["period"]}/predictors/{test_params["framework"]}/{test_params["gcm"]}_{period_inference}.nc'
                lr_ds = xr.open_dataset(lr_path)
                # lr_ds = lr_ds.astype("float32")

        print("loading LR data from:", lr_path)
        
        if data_types_lr is None or data_types_lr == ["all"]:
            # get all available variables except time and lat/lon
            data_types_lr = list(lr_ds.data_vars)
            data_types_lr = sorted(data_types_lr) # ensure same order across different datasets
            # remove time_bounds from predictors list if present
            data_types_lr = [v for v in data_types_lr if v != "time_bnds"]
            
        for data_type in data_types_lr:
            # low-res   
            lr_data_norm = get_normed_data(lr_ds, data_type, norm_input, sqrt_transform_in, root, domain=domain, training_experiment=training_experiment,
                                          gcm_name=gcm_name, period_training=period_training, period_norm=period_norm)
            lr_tensors.append(lr_data_norm.unsqueeze(1))             

        lr_data_allvars = torch.concat(lr_tensors, dim = 1) # shape (n_timesteps, n_vars, spatial_dim), where spatial_dim = 20*36 or 20*36 + value_dim        
        months_np = np.float32(lr_ds.indexes['time'].strftime("%m")).astype("int")
        days_np = np.float32(lr_ds.indexes['time'].strftime("%d")).astype("int")
        years_np = np.float32(lr_ds.indexes['time'].strftime("%Y"))
        time = torch.from_numpy(years_np).unsqueeze(1)
        is_leap = is_leap_year(years_np)
        leap_year_mask = is_leap & (months_np == 2) & (days_np == 29)
        consider_leap = True if np.any(leap_year_mask) else False
        doy = torch.from_numpy(day_of_year_vectorized(months_np, days_np, is_leap, consider_leap=consider_leap)).unsqueeze(1)
        
        # compute the X, Y pairs
        if mode == "train": 
            self.y_data = hr_data_allvars # torch.reshape(hr_data_allvars, (hr_data_allvars.shape[0], -1))
            print("y data:", self.y_data.shape)
            
            self.z_data = hr_data_coarsened_allvars
            print("z (x coarsened) data:", self.z_data.shape)
        
            
        if include_year:
            time_idx = torch.cat([
                    time, # remove time to enable extrapolation
                    doy,
                    torch.sin(torch.tensor(365 / 2 / np.pi) * doy),
                    torch.cos(torch.tensor(365 / 2 / np.pi) * doy),
                    torch.sin(torch.tensor(365 / np.pi) * doy),
                    torch.cos(torch.tensor(365 / np.pi) * doy)
                ], dim = 1)#.expand(lr_data_allvars.shape[0], lr_data_allvars.shape[1], 6)
        else:
            time_idx = torch.cat([
                    # time, # remove time to enable extrapolation
                    doy,
                    torch.sin(torch.tensor(365 / 2 / np.pi) * doy),
                    torch.cos(torch.tensor(365 / 2 / np.pi) * doy),
                    torch.sin(torch.tensor(365 / np.pi) * doy),
                    torch.cos(torch.tensor(365 / np.pi) * doy)
                ], dim = 1)#.expand(lr_data_allvars.shape[0], lr_data_allvars.shape[1], 6)
        
        self.x_data = torch.cat([lr_data_allvars.reshape(lr_data_allvars.shape[0], -1),
                   time_idx],
                    dim = -1)
        print("x data:", self.x_data.shape)
        
        orog_path = root + domain + f"/{domain}_domain/train/{training_experiment}/predictors"
        static_ds = xr.open_dataset(os.path.join(orog_path, orog_file))
        orog_hr = torch.flip(torch.from_numpy(static_ds["orog"].values), [0]).float() # shape (128, 128)
        orog_hr = (orog_hr - orog_hr.mean()) / orog_hr.std() # primitive normalisation
        
        if kernel_size_hr > 1:
            orog_hr =  torch.nn.functional.avg_pool2d(orog_hr.view(1, 1, 128, 128), kernel_size=kernel_size_hr, stride=kernel_size_hr).view(-1)
        else:
            orog_hr = orog_hr.view(-1)
        self.orog = orog_hr # shape (spatial_dim_hr,)
        
    def __len__(self):
        if self.mode == "train":
            assert self.x_data.shape[0] == self.y_data.shape[0]
            assert self.x_data.shape[0] == self.z_data.shape[0]
        if self.return_timepair:
            return self.x_data.shape[0] - 1
        return self.x_data.shape[0]
    
    def __getitem__(self, idx):
        """
        return pre-processed pair
        """
        if self.mode == "inference":
            if self.return_timepair:
                x = self.x_data[idx]
                x_next = self.x_data[idx + 1]
                return x, torch.nan, torch.nan, x_next, torch.nan, torch.nan
            else:
                return self.x_data[idx], torch.nan, torch.nan
        
        elif self.mode == "train":
            if self.return_timepair:
                x = self.x_data[idx]
                z = self.z_data[idx]
                y = self.y_data[idx]
                x_next = self.x_data[idx + 1]
                y_next = self.y_data[idx + 1]
                z_next = self.z_data[idx + 1]
                return x, z, y, x_next, z_next, y_next
            else:
                x = self.x_data[idx]
                y = self.y_data[idx]
                z = self.z_data[idx]
                return x, z, y


# ------------ GET DATA ---------------------------------------

def get_data_cordexbench(
        domain="ALPS",
        training_experiment = 'Emulator_hist_future',
        shuffle=True, batch_size=512,
        tr_te_split = "random", test_size=0.1,
        server="ada",
        variables = ["pr"], variables_lr = None, mode = "train",
        norm_input=None, norm_output=None, sqrt_transform_in=True, sqrt_transform_out=True,
        kernel_size=1, kernel_size_hr=1, return_timepair=False,
        clip_quantile=None, logit=False, normal=False, include_year=False,
        stride_lr=None, padding_lr=None,
        return_dataset=False, period_norm="1961-1980",
        remove_climate_change_trend=False,
        test_params=None, alpha_subtract_linear=100):
    """
    Get data loaders for two-step downscaling (GCM to coarse RCM to fine RCM).
    Wrapper around DownscalingDatasetTwoStepNormed.
    
    Parameters:
    - shuffle: whether to shuffle the training data in the dataloader
    - batch_size: batch size for the dataloader
    - tr_te_split: method to split training and testing data; only "random" is currently included
    - test_size: fraction of data to use as test set
    - server: "ada" or "euler"; determines the root path for data loading
    - variables, variables_lr, mode, norm_input, norm_output, sqrt_transform_in, sqrt_transform_out,
      kernel_size, kernel_size_hr, mask_gcm, return_timepair,
      clip_quantile, logit, normal, include_year,
      only_winter, stride_lr, padding_lr,
      filter_outliers, precip_zeros: parameters passed to DownscalingDatasetTwoStepNormed
    """
    if server == "ada":
        root="/r/scratch/users/mschillinger/data/cordexbench/"
    elif server == "euler":
        raise NotImplementedError("Data not on Euler yet.")

    random_state = 42
    
    if mode != "train":
        test_size = 0.0
        
            
    full_dataset = DownscalingDatasetTwoStepNormed(root = root,
                                            domain=domain,
                                            training_experiment = training_experiment,
                                            data_types=variables, 
                                            data_types_lr = variables_lr,
                                            mode=mode,
                                            norm_input=norm_input,
                                            norm_output=norm_output,
                                            sqrt_transform_in=sqrt_transform_in,
                                            sqrt_transform_out=sqrt_transform_out,
                                            kernel_size=kernel_size,
                                            kernel_size_hr=kernel_size_hr,
                                            return_timepair=return_timepair,
                                            clip_quantile=clip_quantile,
                                            logit=logit,
                                            normal=normal,
                                            include_year=include_year,
                                            stride_lr=stride_lr,
                                            padding_lr=padding_lr,
                                            period_norm=period_norm,
                                            remove_climate_change_trend=remove_climate_change_trend,
                                            test_params=test_params,
                                            alpha_subtract_linear=alpha_subtract_linear)
    
    if tr_te_split == "random":
        # REMOVED all RCMs except ALADIN
        
        if test_size > 0:
            train_indices, test_indices = train_test_split(list(range(len(full_dataset))), test_size = test_size, random_state = random_state)
            dataset_train = Subset(full_dataset, train_indices)
            dataset_test = Subset(full_dataset, test_indices)
            dataloader_train = DataLoader(dataset_train, batch_size, shuffle=shuffle)
            dataloader_test = DataLoader(dataset_test, batch_size, shuffle=shuffle)
        else:
            dataloader_train = DataLoader(full_dataset, batch_size, shuffle=shuffle)
            dataloader_test = None
    elif tr_te_split == "past_future":
        train_indices = np.arange(0, (14600//2))
        test_indices = np.arange(14600//2 + 1, 14600)
        dataset_train = Subset(full_dataset, train_indices)
        dataset_test = Subset(full_dataset, test_indices)
        dataloader_train = DataLoader(dataset_train, batch_size, shuffle=shuffle)
        dataloader_test = DataLoader(dataset_test, batch_size, shuffle=shuffle)    
    else:
        raise NotImplementedError("Please use tr_te_split=random or past_future, other options have been removed for simplicity.")
    
    if return_dataset:
        return dataloader_train, dataloader_test, full_dataset
    return dataloader_train, dataloader_test