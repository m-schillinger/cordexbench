import torch
from modules import *
from modules_loc_variant import *
from modules_orog import *
from utils import *
from data import *
import argparse
import pdb
import time

class args:
    latent_dims = [None, 12, 12, 12, 12]
    preproc_layers = [True, False, False, False, False]
    noise_dims = [10, 5, 5, 5, 5]
    layer_shrinkages = [1, None, None, None, None]
    model_types = ["dense", "nicolai", "nicolai", "nicolai", "nicolai"]
    one_hot_options = [None, "argument", "argument", "argument", "argument"]
    one_hot_flags = [False, False, False, False, False]

    conv_dims = [None, None, None, None, None]
    kernel_sizes = [16, 8, 4, 2, 1]
    vars_as_channels = [False, False, False, False, False]

    hidden_dim_t = 200
    num_layer_t = 6
    preproc_layer_t = True
    preproc_dim_t = 50
    noise_dim_t = 10
    out_act_t = None
    layer_shrinkage_t = 1

    burn_ins = [399, 499, 99, 199]
    # specifications for nicolai layers
    num_neighbors_ups = [None, 9, 9, 9, 9]
    num_neighbors_res = [None, 25, 25, 25, 25]
    noise_dim_mlp = [None, 0, 0, 0, 0]
    batch_size = 128
    n_models = 1
    noise_std  = 1
    bn = False
    mlp = True
    val_dim = None

    approx_unif = False # use approximate backtransformation for uniform for speed up    
    variables_lr =  ["all"]
    n_visual = 5
    save_dir_super = None
    method = "eng_2step"
    avg_constraint = False # True for old runs, but then changed to False for newer version
    logit_transform = False
    sep_mean_std = False
    lambda_coarse = 1 #0.9 vs 1 (coarse from super vs pure coarse)
    norm_loss_batch = True
    norm_loss_loc = True
    server = "ada"
    save_quantiles = False #usually False, but set to true for saving quantiles over many samples for more accurate QL and MSE

    
def get_model(args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, test_loader_in = get_data_cordexbench(
        domain=args.domain,
        training_experiment=args.training_experiment,
        shuffle=True, batch_size=512,
        tr_te_split = "random", test_size=0.1,
        server=args.server,
        variables=args.variables, variables_lr=args.variables_lr,
        mode = "train",
        norm_input=args.norm_method_input, norm_output=args.norm_method_output,
        sqrt_transform_in=args.sqrt_transform_in, sqrt_transform_out=args.sqrt_transform_out,
        kernel_size=args.kernel_sizes[0], kernel_size_hr=1, return_timepair=False,
        clip_quantile=None, 
        logit=args.logit_transform,
        normal=args.normal_transform,
        include_year=False,
        stride_lr=None, padding_lr=None,
        )
    
    x_tr_eval, xc_tr_eval, y_tr_eval = next(iter(train_loader))
    x_tr_eval, xc_tr_eval, y_tr_eval = x_tr_eval[:args.n_visual].to(device), xc_tr_eval[:args.n_visual].to(device), y_tr_eval[:args.n_visual].to(device)
    x_te_eval, xc_te_eval, y_te_eval = next(iter(test_loader_in))
    x_te_eval, xc_te_eval, y_te_eval = x_te_eval[:args.n_visual].to(device), xc_te_eval[:args.n_visual].to(device), y_te_eval[:args.n_visual].to(device)
    
    if args.variables_lr is not None:
        n_vars = len(args.variables_lr)
    else:
        n_vars = 5
    assert args.norm_method_output != "rank_val"
    in_dim = x_tr_eval.shape[1]
    
    #### get norm stats file (aligned with utils.normalise/unnormalise)
    norm_stats = {}
    # derive GCM and training period consistent with dataset logic
    if args.domain == "ALPS":
        gcm_name = "CNRM-CM5"
    elif args.domain == "NZ" or args.domain == "SA":
        gcm_name = "ACCESS-CM2"
    else:
        raise ValueError("Unsupported domain for norm stats: " + str(args.domain))

    if args.training_experiment == "ESD_pseudo_reality":
        period_training = "1961-1980"
    elif args.training_experiment == "Emulator_hist_future":
        period_training = "1961-1980_2080-2099"
    else:
        raise ValueError("Unsupported training_experiment for norm stats: " + str(args.training_experiment))

    for var in args.variables:
        mode_unnorm = "hr"
        name_str = "_sqrt" if (var in ["pr", "sfcWind"] and args.sqrt_transform_out) else ""
        file_base = f"{args.training_experiment}_{var}_{gcm_name}_{period_training}{name_str}"

        if args.norm_method_output == "normalise_pw" or args.norm_method_output == "scale_pw":
            ns_path = os.path.join(args.data_dir, "norm_stats", f"{mode_unnorm}_norm_stats_pixelwise_{file_base}.pt")
            norm_stats[var] = torch.load(ns_path, map_location=device)
        elif args.norm_method_output == "normalise_scalar":
            ns_path = os.path.join(args.data_dir, "norm_stats", f"{mode_unnorm}_norm_stats_full-data_{file_base}.pt")
            norm_stats[var] = torch.load(ns_path, map_location=device)
        elif args.norm_method_output == "normalise_per_period":
            if args.period_norm == "1961-1980":
                name = "past"
            elif args.period_norm == "2080-2099":
                name = "future-estimated"
            ns_path = os.path.join(args.data_dir, "norm_stats",
                                   f"{mode_unnorm}_norm_stats_full-data_{name}_{file_base}.pt")
            print("loading norm stats from", ns_path)
            norm_stats[var] = torch.load(ns_path, weights_only=False)
        elif args.norm_method_output == "subtract_linear" and args.mode == "train":      
            ns_path = os.path.join(args.data_dir, "norm_stats", f"{mode_unnorm}_norm_stats_linear-pred_all_{file_base}.pt")
            norm_stats[var] = torch.load(ns_path)
        elif args.norm_method_output == "subtract_linear" and args.mode == "inference":
            norm_stats[var] = None # need to load in eval loop to make sure aligned with test params (e.g. period norm, alpha value)
            #ns_path = os.path.join(
            #        "/r/scratch/groups/nm/downscaling/samples_cordexbench/", args.training_experiment, args.domain, "no_orog", var, "linear_pred",
            #        f"linear-pred_{test_params['period']}_{test_params['framework']}_{test_params['gcm']}_alpha-{args.alpha_subtract_linear}.pt")
            #norm_stats[var] = torch.load(ns_path)
        else:
            norm_stats[var] = None
            
    in_dim = x_tr_eval.shape[1]
        
    if args.server == "euler":
        # prefix = "/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear/results/"
        pass
    elif args.server == "ada":
        if args.add_orography:
            prefix = f"results/{args.training_experiment}/{args.domain}/with-orog/"
        else:
            prefix = f"results/{args.training_experiment}/{args.domain}/no-orog/"
        prefix0 = f"results/{args.training_experiment}/{args.domain}/no-orog/" # coarse model always here
                   
    
    loaded_models = []
    for i in range(len(args.model_dirs)):
        num_layer = args.num_layers[i]
        hidden_dim = args.hidden_dims[i]
        preproc_layer = args.preproc_layers[i]
        noise_dim = args.noise_dims[i]
        layer_shrinkage = args.layer_shrinkages[i]
        out_act = args.out_acts[i]
        one_hot = args.one_hot_flags[i]
        burn_in = args.burn_ins[i]
        model_dir = args.model_dirs[i]

        if i == 0:
            save_dir = prefix0 + model_dir
        else:
            save_dir = prefix + model_dir

        if args.model_types[i] == "dense":
            if i == 0:
                in_dim_model = in_dim
            else:
                in_dim_model = out_dim_model
                if args.one_hot_flags[i]:
                    in_dim_model += 7
            
            out_dim_model = int(128 / args.kernel_sizes[i])**2 * len(args.variables)
            print("parameters in model", in_dim_model, out_dim_model)
            print(num_layer, hidden_dim, noise_dim, preproc_layer, layer_shrinkage)
            
            if args.variables_lr == ["all"]:
                input_dims_for_preproc = np.array([256 for k in range(15)] +
                        [5])
            else:
                input_dims_for_preproc = np.array([256 for k in range(len(args.variables_lr))] +
                 [5])
            
            model = StoUNet(
                in_dim_model,
                out_dim_model,
                num_layer,
                hidden_dim,
                noise_dim=noise_dim,
                add_bn=args.bn, out_act=out_act, 
                resblock=args.mlp, noise_std=args.noise_std,
                preproc_layer=args.preproc_layer_t, 
                input_dims_for_preproc=input_dims_for_preproc,
                preproc_dim=args.preproc_dims[i],
                layer_shrinkage=layer_shrinkage,
            ).to(device)
            
        elif args.model_types[i] == "conv":
            # TO DO: make sure 4x resolution jump
            # TO DO: image size
             model = Generator4xExternalNoise(
                conv_dim=args.conv_dims[i],
                image_size=32,
                n_channels=len(args.variables),
                one_hot_channel=one_hot,
                one_hot_dim=7 if one_hot else None
            ).to(device)
    
        elif args.model_types[i] == "nicolai":
            assert i > 0
            
            if not args.add_orography:
                print(in_dim_model, out_dim_model)
            
                num_classes = 1
                model = RectUpsampleWithResiduals(int(128 / args.kernel_sizes[i-1]),
                                int(128 / args.kernel_sizes[i]),
                                n_features=len(args.variables),
                                num_classes=num_classes, 
                                num_neighbors_ups=args.num_neighbors_ups[i],
                                num_neighbors_res=args.num_neighbors_res[i],
                                map_dim=args.latent_dims[i],
                                noise_dim=noise_dim,
                                mlp_hidden=hidden_dim,
                                mlp_depth=num_layer,
                                noise_dim_mlp=args.noise_dim_mlp[i],
                                double_linear=args.double_linear[i],
                                split_residuals=not args.not_split_residuals,
                                out_act=out_act,
                                ).to(device)
    
            else:
                num_classes = 1
                model = UpsampleWithOrography(128//args.kernel_sizes[i-1],
                            128 // args.kernel_sizes[i],
                            n_features=len(args.variables),
                            num_classes=num_classes,
                            num_neighbors_orog=args.num_neighbors_res[i],
                            num_neighbors_ups=args.num_neighbors_ups[i],
                            num_neighbors_res=args.num_neighbors_res[i],
                            orog_latent_dim=3,
                            orog_input_dim=1,
                            map_dim=args.latent_dims[i],
                            noise_dim=noise_dim,
                            mlp_hidden=hidden_dim,
                            mlp_depth=num_layer,
                            noise_dim_mlp=args.noise_dim_mlp[i],
                            split_residuals=not args.not_split_residuals,
                            orog_nonlinear=True,
                            out_act=out_act,
                            ).to(device)    
        ckpt_path = os.path.join(save_dir, f"model_{burn_in}.pt")
        print(f"Loading model from {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        loaded_models.append(model)
    
    if not args.temporal:
        
        model_spec_list = \
            [
            ModelSpec(
                model = loaded_models[i],
                vars_as_channels = args.vars_as_channels[i],
                use_one_hot = args.one_hot_flags[i],
                noise_dim = args.noise_dims[i] * int(128 / args.kernel_sizes[i])**2, # latter part is the number of pixels for which we need noise
                one_hot_option = args.one_hot_options[i],
                include_orog = args.add_orography and args.model_types[i] == "nicolai"
            )
            for i in range(len(loaded_models))
            ]
        sequential_model = HierarchicalWrapper(model_spec_list, n_vars=len(args.variables), one_hot_dim=7)
        coarse_model_marginal = loaded_models[0]
    else:
        # load coarse temporal model
        out_dim_model = int(128 / args.kernel_sizes[0])**2 * len(args.variables)
        model_temp = StoUNet(
                in_dim,
                out_dim_model,
                args.num_layer_t,
                args.hidden_dim_t,
                noise_dim=args.noise_dim_t,
                add_bn=args.bn, out_act=args.out_act_t, 
                resblock=args.mlp, noise_std=args.noise_std,
                preproc_layer=args.preproc_layer_t, 
                input_dims_for_preproc=np.array(
                                            [720  for k in range(n_vars)] +
                                            [5, 7] +
                                            [out_dim_model // len(args.variables) for k in range(len(args.variables))]),
                preproc_dim=args.preproc_dim_t,
                layer_shrinkage=args.layer_shrinkage_t
            ).to(device)
        save_dir_t = prefix + args.model_dir_t
        ckpt_path = os.path.join(save_dir_t, f"model_{args.burn_in_t}.pt")
        print(f"Loading model from {ckpt_path}")
        model_temp.load_state_dict(torch.load(ckpt_path, map_location=device))
        model_temp.eval()
        
        model_spec_list = \
            [ModelSpec(
                model = model_temp,
                vars_as_channels = False,
                use_one_hot = False, # because one hot doesn't need to be concatenated anymore
                noise_dim = args.noise_dim_t,
            )    
            ] + \
            [ # load remaining models
            ModelSpec(
                model = loaded_models[i],
                vars_as_channels = args.vars_as_channels[i],
                use_one_hot = args.one_hot_flags[i],
                noise_dim = args.noise_dims[i] * int(128 / args.kernel_sizes[i])**2,
                one_hot_option = args.one_hot_options[i],
            )
            for i in range(1, len(loaded_models))
            ]
        sequential_model = HierarchicalWrapper(model_spec_list, n_vars=len(args.variables), one_hot_dim=7)
        coarse_model_marginal = loaded_models[0]
    
    param_dict = {
        "save_dir": save_dir,
        "device": device,
        "norm_stats": norm_stats
    }
    if args.temporal:
        param_dict["save_dir_t"] = save_dir_t
        param_dict["burn_in_t"] = args.burn_in_t
        
    
    return sequential_model, coarse_model_marginal, param_dict

if __name__ == '__main__':
    # Load the model
    parser = argparse.ArgumentParser(description='Evaluate three-step coarse from super model')
    parser.add_argument('--temporal', action='store_true', help='Enable temporal mode')
    #counterfactuals = False #usually False, only for extra experiment True
    #split_coarse_super = False #usually False, only for extra experiment True
    #pure_super = False #usually False, only for extra experiment True
    parser.add_argument('--version', type=str, default = "6", help='Version of the samples to save')
    parser.add_argument('--mode', type=str, choices = ["train", "inference"], default="train")
    parser.add_argument('--norm_option', type=str, choices = ["none", "pw", "scale_pw", "scalar_in", "per_period", "subtract_linear"])
    parser.add_argument('--add_interm_loss', action='store_true', help='Enable intermediate loss')
    parser.add_argument('--add_mse_loss', action='store_true', help='Enable extra MSE loss')
    parser.add_argument('--use_double_linear', action='store_true')
    parser.add_argument('--nicolai_layers', action='store_true', help='Enable nicolai layers')
    parser.add_argument('--precip_zeros', type=str, default="random")
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay in optimisers')
    parser.add_argument('--domain', type=str, default="ALPS", help='domain to use')
    parser.add_argument('--variable', type=str, default="tasmax", help='variable to use')
    parser.add_argument('--training_experiment', type=str, default="Emulator_hist_future")
    parser.add_argument('--add_orography', action='store_true', help='Enable adding orography')
    parser.add_argument('--past_future', action='store_true', help='Take models only trained on past')
    parser.add_argument('--period_norm', choices=["past", "future"], default="past")
    parser.add_argument('--remove_climate_change_trend', action='store_true', help='Enable removing climate change trend for evaluation')
    parser.add_argument('--alpha_subtract_linear', type=float, default=100, help='Alpha value for subtract_linear normalization')

    args_parsed = parser.parse_args()
    args.temporal = args_parsed.temporal
    args.version = args_parsed.version
    args.add_interm_loss = args_parsed.add_interm_loss
    args.norm_option = args_parsed.norm_option
    args.nicolai_layers = args_parsed.nicolai_layers
    args.add_mse_loss = args_parsed.add_mse_loss
    args.use_double_linear = args_parsed.use_double_linear
    args.precip_zeros = args_parsed.precip_zeros
    args.weight_decay = args_parsed.weight_decay
    args.domain = args_parsed.domain
    args.variable = args_parsed.variable
    args.variables = [args.variable]
    args.training_experiment = args_parsed.training_experiment
    args.add_orography = args_parsed.add_orography
    args.past_future = args_parsed.past_future
    args.mode = args_parsed.mode
    args.alpha_subtract_linear = args_parsed.alpha_subtract_linear
    orog_folder = "with-orog" if args.add_orography else "no-orog"
    if args_parsed.period_norm == "past":
        args.period_norm = "1961-1980"
    elif args_parsed.period_norm == "future":
        args.period_norm = "2080-2099"
    args.remove_climate_change_trend = args_parsed.remove_climate_change_trend
    # TO DO: update
    if args.server == "ada":
        args.data_dir = "/r/scratch/users/mschillinger/data/cordexbench/" + args.domain
    elif args.server == "euler":
        pass

    if args.add_interm_loss:
        args.latent_dims = [None, 4, 4, 4, 4] # 4 if loss intermediate
    else:
        args.latent_dims = [None, 12, 12, 12, 12]

    if args.norm_option == "none":
        args.norm_method_input = None
        args.norm_method_output = None
        args.normal_transform = False # TO DO: correct?; check for temperature
        args.sqrt_transform_in = False
        args.sqrt_transform_out = False
    elif args.norm_option == "scalar_in":
        args.norm_method_input = "normalise_scalar"
        args.norm_method_output = None
        args.normal_transform = False
        args.sqrt_transform_in = True
        args.sqrt_transform_out = True
    elif args.norm_option == "pw":
        args.norm_method_input = "normalise_scalar"
        args.norm_method_output = "normalise_pw"
        args.normal_transform = False
        args.sqrt_transform_in = True
        args.sqrt_transform_out = True
    elif args.norm_option == "scale_pw":
        args.norm_method_input = "normalise_scalar"
        args.norm_method_output = "scale_pw"
        args.normal_transform = False
        args.sqrt_transform_in = True
        args.sqrt_transform_out = True
    elif args.norm_option == "per_period":
        args.norm_method_input = "normalise_per_period"
        args.norm_method_output = "normalise_per_period"
        args.normal_transform = False
        args.sqrt_transform_in = True
        args.sqrt_transform_out = True
    elif args.norm_option == "subtract_linear":
        args.norm_method_input = "normalise_scalar"
        args.norm_method_output = "subtract_linear"
        args.normal_transform = False
        args.sqrt_transform_in = True
        args.sqrt_transform_out = True

    if args.nicolai_layers:
        args.num_layers = [6, 2, 2, 2, 2]
        args.hidden_dims = [200, 12, 12, 12, 12]
        args.preproc_layers = [True, False, False, False, False]
        args.preproc_dims = [20, None, None, None, None]
        args.noise_dims = [10, 5, 5, 5, 5]
        args.layer_shrinkages = [1, None, None, None, None]        
        
        args.model_types = ["dense", "nicolai", "nicolai", "nicolai", "nicolai"]
        args.one_hot_options = [None, "argument", "argument", "argument", "argument"]
        args.conv_dims = [None, None, None, None, None]
        args.kernel_sizes = [16, 8, 4, 2, 1]
        args.vars_as_channels = [False, False, False, False, False]
        args.one_hot_flags = [False, False, False, False, False]
        
        # specifications for nicolai layers
        args.num_neighbors_ups = [None, 9, 9, 9, 9]
        
        noise_dim_mlp = [None, 0, 0, 0, 0]
        if args.use_double_linear:
            args.double_linear = [None, True, True, True, True]
        else:
            args.double_linear = [None, False, False, False, False]        
        
        # --- common options -------------------
        if args.norm_option == "pw" and args.variable == "tasmax" and not args.past_future:
            args.out_acts = [None, None, None, None, None]
            args.latent_dims = [None, 12, 12, 12, 12] 
            args.num_neighbors_res = [None, 25, 25, 25, 25]
            args.model_dirs = [
                #"coarse/var-tas_pr_sfcWind_rsds/hd-200_num-lay-6_norm-out-normalise_pw_preproc_norm-loss-per-var-p4/",
                "coarse/var-tasmax/hd-100_num-lay-6_norm-out-normalise_pw_norm-in-normalise_scalar_nd-100_sorted-pred/",
                "super/lr16_hr8/var-tasmax/loc-specific-layers_norm-out-normalise_pw_dec0_lam-mse0_split-residTrue/",
                "super/lr8_hr4/var-tasmax/loc-specific-layers_norm-out-normalise_pw_dec0_lam-mse0_split-residTrue/",
                "super/lr4_hr2/var-tasmax/loc-specific-layers_norm-out-normalise_pw_dec0_lam-mse0_split-residTrue/",
                "super/lr2_hr1/var-tasmax/loc-specific-layers_norm-out-normalise_pw_dec0_lam-mse0_split-residTrue/",
            ]
            args.burn_ins = [999, 3999, 3999, 1999, 999]
            args.not_split_residuals = False
            args.hidden_dims[0] = 100
            args.noise_dims[0] = 100
            
        elif args.norm_option == "subtract_linear" and args.variable == "tasmax" and not args.past_future:
            args.out_acts = [None, None, None, None, None]
            args.latent_dims = [None, 12, 12, 12, 12] 
            args.num_neighbors_res = [None, 25, 25, 25, 25]
            if args.alpha_subtract_linear != 100:
                alpha_str = f"_alpha-{int(args.alpha_subtract_linear)}"
            else:
                alpha_str = ""
            args.model_dirs = [
                #"coarse/var-tas_pr_sfcWind_rsds/hd-200_num-lay-6_norm-out-normalise_pw_preproc_norm-loss-per-var-p4/",
                f"coarse/var-tasmax/hd-100_num-lay-6_norm-out-subtract_linear_norm-in-normalise_scalar_nd-100{alpha_str}_sorted-pred/",
                f"super/lr16_hr8/var-tasmax/loc-specific-layers_norm-out-subtract_linear_dec0_lam-mse0_split-residTrue{alpha_str}/",
                f"super/lr8_hr4/var-tasmax/loc-specific-layers_norm-out-subtract_linear_dec0_lam-mse0_split-residTrue{alpha_str}/",
                f"super/lr4_hr2/var-tasmax/loc-specific-layers_norm-out-subtract_linear_dec0_lam-mse0_split-residTrue{alpha_str}/",
                f"super/lr2_hr1/var-tasmax/loc-specific-layers_norm-out-subtract_linear_dec0_lam-mse0_split-residTrue{alpha_str}/",
            ]
            args.burn_ins = [999, 3999, 3999, 1999, 999]
            args.not_split_residuals = False
            args.hidden_dims[0] = 100
            args.noise_dims[0] = 100

        elif args.norm_option == "scale_pw" and args.variable == "pr" and not args.past_future and args.weight_decay == 0:
            args.out_acts = ["relu", "relu", "relu", "relu", "relu"]
            args.num_neighbors_res = [None, 25, 25, 25, 25]
            args.model_dirs = [
                #"coarse/var-tas_pr_sfcWind_rsds/hd-200_num-lay-6_norm-out-normalise_pw_preproc_norm-loss-per-var-p4/",
                "coarse/var-pr/hd-100_num-lay-6_norm-out-scale_pw_relu_norm-in-normalise_scalar_nd-100_sorted-pred/",
                "super/lr16_hr8/var-pr/loc-specific-layers_norm-out-scale_pw_relu_dec0_lam-mse0_split-residTrue/",
                "super/lr8_hr4/var-pr/loc-specific-layers_norm-out-scale_pw_relu_dec0_lam-mse0_split-residTrue/",
                "super/lr4_hr2/var-pr/loc-specific-layers_norm-out-scale_pw_relu_dec0_lam-mse0_split-residTrue/",
                "super/lr2_hr1/var-pr/loc-specific-layers_norm-out-scale_pw_relu_dec0_lam-mse0_split-residTrue/",
            ]
            args.burn_ins = [999, 3999, 3999, 1999, 999]
            args.not_split_residuals = False
            args.hidden_dims[0] = 100
            args.noise_dims[0] = 100
        
        elif args.norm_option == "subtract_linear" and args.variable == "pr" and not args.past_future and args.weight_decay == 0:
            args.out_acts = [None, None, None, None, None]
            args.num_neighbors_res = [None, 25, 25, 25, 25]
            if args.alpha_subtract_linear != 100:
                alpha_str = f"_alpha-{int(args.alpha_subtract_linear)}"
            else:
                alpha_str = ""
            args.model_dirs = [
                #"coarse/var-tas_pr_sfcWind_rsds/hd-200_num-lay-6_norm-out-normalise_pw_preproc_norm-loss-per-var-p4/",
                f"coarse/var-pr/hd-100_num-lay-6_norm-out-subtract_linear_norm-in-normalise_scalar_nd-100{alpha_str}_sorted-pred/",
                f"super/lr16_hr8/var-pr/loc-specific-layers_norm-out-subtract_linear_dec0_lam-mse0_split-residTrue{alpha_str}/",
                f"super/lr8_hr4/var-pr/loc-specific-layers_norm-out-subtract_linear_dec0_lam-mse0_split-residTrue{alpha_str}/",
                f"super/lr4_hr2/var-pr/loc-specific-layers_norm-out-subtract_linear_dec0_lam-mse0_split-residTrue{alpha_str}/",
                f"super/lr2_hr1/var-pr/loc-specific-layers_norm-out-subtract_linear_dec0_lam-mse0_split-residTrue{alpha_str}/",
            ]
            args.burn_ins = [499, 3999, 3999, 1999, 999]
            args.not_split_residuals = False
            args.hidden_dims[0] = 100
            args.noise_dims[0] = 100
            
        # ------------- archive of options --------------
        elif args.norm_option == "scalar_in" and args.variable == "pr":
            args.out_acts = ["relu", None, None, None, None]
            args.latent_dims = [None, 12, 12, 12, 12] 
            args.num_neighbors_res = [None, 9, 9, 25, 25] # BUG
            args.model_dirs = [
                #"coarse/var-tas_pr_sfcWind_rsds/hd-200_num-lay-6_norm-out-normalise_pw_preproc_norm-loss-per-var-p4/",
                "coarse/var-pr/hd-200_num-lay-6_norm-out-None_relu_norm-in-normalise_scalar/",
                "super/lr16_hr8/var-pr/loc-specific-layers_norm-out-None_dec0_lam-mse0_split-residTrue/",
                "super/lr8_hr4/var-pr/loc-specific-layers_norm-out-None_dec0_lam-mse0_split-residTrue/",
                "super/lr4_hr2/var-pr/loc-specific-layers_norm-out-None_dec0_lam-mse0_split-residTrue/",
                "super/lr2_hr1/var-pr/loc-specific-layers_norm-out-None_dec0_lam-mse0_split-residTrue/",
            ]
            args.burn_ins = [999, 3999, 3999, 999, 999]
            args.not_split_residuals = False
            
        elif args.norm_option == "scale_pw" and args.variable == "pr" and not args.past_future and args.weight_decay == 1e-3:
            args.out_acts = ["relu", "relu", "relu", "relu", "relu"]
            args.num_neighbors_res = [None, 25, 25, 25, 25]
            args.model_dirs = [
                #"coarse/var-tas_pr_sfcWind_rsds/hd-200_num-lay-6_norm-out-normalise_pw_preproc_norm-loss-per-var-p4/",
                "coarse/var-pr/hd-100_num-lay-6_norm-out-scale_pw_relu_norm-in-normalise_scalar_nd-100/",
                "super/lr16_hr8/var-pr/loc-specific-layers_norm-out-scale_pw_relu_dec1e-3_lam-mse0_split-residFalse/",
                "super/lr8_hr4/var-pr/loc-specific-layers_norm-out-scale_pw_relu_dec1e-3_lam-mse0_split-residFalse/",
                "super/lr4_hr2/var-pr/loc-specific-layers_norm-out-scale_pw_relu_dec1e-3_lam-mse0_split-residFalse/",
                "super/lr2_hr1/var-pr/loc-specific-layers_norm-out-scale_pw_relu_dec1e-3_lam-mse0_split-residFalse/",
            ]
            args.burn_ins = [999, 3999, 3999, 1999, 999]
            args.not_split_residuals = True
            args.hidden_dims[0] = 100
            args.noise_dims[0] = 100
            
        # ---------- past future options to test extrapolation ------------
        
        
        elif args.norm_option == "subtract_linear" and args.variable == "tasmax" and args.past_future:
            print("test training based only on past data")
            args.out_acts = [None, None, None, None, None]
            args.latent_dims = [None, 12, 12, 12, 12] 
            args.num_neighbors_res = [None, 9, 9, 25, 25] # BUG
            args.model_dirs = [
                #"coarse/var-tas_pr_sfcWind_rsds/hd-200_num-lay-6_norm-out-normalise_pw_preproc_norm-loss-per-var-p4/",
                # "coarse/var-tasmax/hd-100_num-lay-6_norm-out-subtract_linear_norm-in-normalise_scalar_nd-100_PAST-FUTURE-SPLIT/",
                "coarse/var-tasmax/hd-100_num-lay-6_norm-out-subtract_linear_norm-in-normalise_scalar-detrended_nd-100_PAST-FUTURE-SPLIT/",
                "super/lr16_hr8/var-tasmax/loc-specific-layers_norm-out-subtract_linear_dec0_lam-mse0_split-residTrue_PAST-FUTURE-SPLIT/",
                "super/lr8_hr4/var-tasmax/loc-specific-layers_norm-out-subtract_linear_dec0_lam-mse0_split-residTrue_PAST-FUTURE-SPLIT/",
                "super/lr4_hr2/var-tasmax/loc-specific-layers_norm-out-subtract_linear_dec0_lam-mse0_split-residTrue_PAST-FUTURE-SPLIT/",
                "super/lr2_hr1/var-tasmax/loc-specific-layers_norm-out-subtract_linear_dec0_lam-mse0_split-residTrue_PAST-FUTURE-SPLIT/",
            ]
            args.burn_ins = [999, 999, 999, 999, 999]
            args.not_split_residuals = False
            args.hidden_dims[0] = 100
            args.noise_dims[0] = 100
            
        elif args.norm_option == "scale_pw" and args.variable == "pr" and args.past_future:
            args.out_acts = ["relu", "relu", "relu", "relu", "relu"]
            args.num_neighbors_res = [None, 9, 9, 25, 25] # BUG
            args.model_dirs = [
                #"coarse/var-tas_pr_sfcWind_rsds/hd-200_num-lay-6_norm-out-normalise_pw_preproc_norm-loss-per-var-p4/",
                "coarse/var-pr/hd-100_num-lay-6_norm-out-scale_pw_relu_norm-in-normalise_scalar_nd-100_PAST-FUTURE-SPLIT/",
                "super/lr16_hr8/var-pr/loc-specific-layers_norm-out-scale_pw_relu_dec0_lam-mse0_split-residTrue_PAST-FUTURE-SPLIT/",
                "super/lr8_hr4/var-pr/loc-specific-layers_norm-out-scale_pw_relu_dec0_lam-mse0_split-residTrue_PAST-FUTURE-SPLIT/",
                "super/lr4_hr2/var-pr/loc-specific-layers_norm-out-scale_pw_relu_dec0_lam-mse0_split-residTrue_PAST-FUTURE-SPLIT/",
                "super/lr2_hr1/var-pr/loc-specific-layers_norm-out-scale_pw_relu_dec0_lam-mse0_split-residTrue_PAST-FUTURE-SPLIT/",
            ]
            args.burn_ins = [699, 3999, 3999, 999, 999]
            args.not_split_residuals = False
            args.hidden_dims[0] = 100
            args.noise_dims[0] = 100

        elif args.norm_option == "subtract_linear" and args.variable == "pr" and args.past_future:
            args.out_acts = [None, None, None, None, None]
            args.num_neighbors_res = [None, 9, 9, 25, 25] # BUG
            args.model_dirs = [
                #"coarse/var-tas_pr_sfcWind_rsds/hd-200_num-lay-6_norm-out-normalise_pw_preproc_norm-loss-per-var-p4/",
                "coarse/var-pr/hd-100_num-lay-6_norm-out-subtract_linear_norm-in-normalise_scalar-detrended_nd-100_PAST-FUTURE-SPLIT/",
                "super/lr16_hr8/var-pr/loc-specific-layers_norm-out-subtract_linear_dec0_lam-mse0_split-residTrue_PAST-FUTURE-SPLIT/",
                "super/lr8_hr4/var-pr/loc-specific-layers_norm-out-subtract_linear_dec0_lam-mse0_split-residTrue_PAST-FUTURE-SPLIT/",
                "super/lr4_hr2/var-pr/loc-specific-layers_norm-out-subtract_linear_dec0_lam-mse0_split-residTrue_PAST-FUTURE-SPLIT/",
                "super/lr2_hr1/var-pr/loc-specific-layers_norm-out-subtract_linear_dec0_lam-mse0_split-residTrue_PAST-FUTURE-SPLIT/",
            ]
            args.burn_ins = [999, 3999, 3999, 999, 399]
            args.not_split_residuals = False
            args.hidden_dims[0] = 100
            args.noise_dims[0] = 100
            
        
        
        elif args.norm_option == "pw" and args.variable == "tasmax" and args.past_future:
            print("test training based only on past data")
            args.out_acts = [None, None, None, None, None]
            args.latent_dims = [None, 12, 12, 12, 12] 
            args.num_neighbors_res = [None, 9, 9, 25, 25] # BUG
            args.model_dirs = [
                #"coarse/var-tas_pr_sfcWind_rsds/hd-200_num-lay-6_norm-out-normalise_pw_preproc_norm-loss-per-var-p4/",
                "coarse/var-tasmax/hd-100_num-lay-6_norm-out-normalise_pw_norm-in-normalise_scalar_nd-100_PAST-FUTURE-SPLIT/",
                "super/lr16_hr8/var-tasmax/loc-specific-layers_norm-out-normalise_pw_dec0_lam-mse0_split-residTrue_PAST-FUTURE-SPLIT/",
                "super/lr8_hr4/var-tasmax/loc-specific-layers_norm-out-normalise_pw_dec0_lam-mse0_split-residTrue_PAST-FUTURE-SPLIT/",
                "super/lr4_hr2/var-tasmax/loc-specific-layers_norm-out-normalise_pw_dec0_lam-mse0_split-residTrue_PAST-FUTURE-SPLIT/",
                "super/lr2_hr1/var-tasmax/loc-specific-layers_norm-out-normalise_pw_dec0_lam-mse0_split-residTrue_PAST-FUTURE-SPLIT/",
            ]
            args.burn_ins = [899, 3999, 3999, 999, 999]
            args.not_split_residuals = False
            args.hidden_dims[0] = 100
            args.noise_dims[0] = 100
            
        elif args.norm_option == "per_period" and args.variable == "tasmax" and args.past_future:
            print("test training based only on past data")
            args.out_acts = [None, None, None, None, None]
            args.latent_dims = [None, 12, 12, 12, 12] 
            args.num_neighbors_res = [None, 9, 9, 25, 25] # BUG
            args.model_dirs = [
                #"coarse/var-tas_pr_sfcWind_rsds/hd-200_num-lay-6_norm-out-normalise_pw_preproc_norm-loss-per-var-p4/",
                "coarse/var-tasmax/hd-100_num-lay-6_norm-out-normalise_per_period_norm-in-normalise_per_period_nd-100_PAST-FUTURE-SPLIT/",
                "super/lr16_hr8/var-tasmax/loc-specific-layers_norm-out-normalise_per_period_dec0_lam-mse0_split-residTrue_PAST-FUTURE-SPLIT/",
                "super/lr8_hr4/var-tasmax/loc-specific-layers_norm-out-normalise_per_period_dec0_lam-mse0_split-residTrue_PAST-FUTURE-SPLIT/",
                "super/lr4_hr2/var-tasmax/loc-specific-layers_norm-out-normalise_per_period_dec0_lam-mse0_split-residTrue_PAST-FUTURE-SPLIT/",
                "super/lr2_hr1/var-tasmax/loc-specific-layers_norm-out-normalise_per_period_dec0_lam-mse0_split-residTrue_PAST-FUTURE-SPLIT/",
            ]
            args.burn_ins = [999, 999, 999, 99, 99]
            args.not_split_residuals = False
            args.hidden_dims[0] = 100
            args.noise_dims[0] = 100
            

        
        else:
            raise ValueError("Model directory not specified for this configuration")
    model, coarse_model_marginal, param_dict = get_model(args)
    
    device = param_dict["device"]
    norm_stats = param_dict["norm_stats"]
    
    if args.temporal:
        save_dir_t = param_dict["save_dir_t"]
        burn_in_t = param_dict["burn_in_t"]


    if args.server == "euler":
        save_dir_samples = "/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear/samples_cordexbench/" + f"maybritt_{args.version}/"
    else:
        save_dir_samples = f"/r/scratch/groups/nm/downscaling/samples_cordexbench/{args.training_experiment}/{args.domain}/{orog_folder}/{args.variable}/" + f"maybritt_{args.version}/"   
    
    print("saving samples in ", save_dir_samples)
    os.makedirs(save_dir_samples, exist_ok=True)

    with open(os.path.join(save_dir_samples, "ckpt_dirs.txt"), "w") as f:
        for i in range(len(args.model_dirs)):
            f.write(f"Model {i} checkpoint: {os.path.join(args.model_dirs[i], f'model_{args.burn_ins[i]}.pt')}\n")
    
        if args.temporal:
            f.write(f"Saving temporal model: {os.path.join(args.model_dir_t, f'model_{args.burn_in_t}.pt')}\n")
    
    
    # run_indices = np.arange(0,8)
    mode_unnorm = "hr"
    modes = [args_parsed.mode]
    counterfactuals = False
    
    # pass to norm stats for completenes (but not needed for inference)
    if args.domain == "ALPS":
        gcm_name = "CNRM-CM5"
    elif args.domain == "NZ" or args.domain == "SA":
        gcm_name = "ACCESS-CM2"
    else:
        raise ValueError("Unsupported domain for norm stats: " + str(args.domain))

    if args.training_experiment == "ESD_pseudo_reality":
        period_training = "1961-1980"
    elif args.training_experiment == "Emulator_hist_future":
        period_training = "1961-1980_2080-2099"
    else:
        raise ValueError("Unsupported training_experiment for norm stats: " + str(args.training_experiment))
    
    test_periods = ["historical", "mid_century", "end_century"]
    frameworks = ["perfect", "imperfect"]
    if args.domain == "ALPS":
        gcms = ["MPI-ESM-LR", "CNRM-CM5"]
    elif args.domain == "NZ":
        gcms = ["ACCESS-CM2", "EC-Earth3"]
    elif args.domain == "SA":
        gcms = ["ACCESS-CM2", "NorESM2-MM"]
    
    for mode in modes:
        print(f"--- Processing Mode: {mode} ---")
        
        # 1. Determine iterations
        # If mode is train, we run once. If test, we loop through params.
        configs = [{}] # Default empty for train
        if mode == "inference":
            configs = [
                {"period": p, "framework": f, "gcm": g}
                for p in test_periods for f in frameworks for g in gcms
            ]

        for config in configs:
            if mode == "inference":
                print(f"Running Inference: {config['period']} | {config['framework']} | {config['gcm']}")
                test_params = config
                period_norm = None
            else:
                if args_parsed.period_norm == "future":
                    period_norm = "2080-2099"
                else:
                    period_norm = "1961-1980"
                test_params = None

            test_loader_in, _, full_dataset = get_data_cordexbench(
                domain=args.domain,
                training_experiment=args.training_experiment,
                shuffle=False, batch_size=512,
                tr_te_split = "random", test_size=0.0,
                server=args.server,
                variables=args.variables, variables_lr=args.variables_lr,
                mode = mode,
                norm_input=args.norm_method_input, norm_output=args.norm_method_output,
                sqrt_transform_in=args.sqrt_transform_in, sqrt_transform_out=args.sqrt_transform_out,
                kernel_size=args.kernel_sizes[0], kernel_size_hr=1, return_timepair=False,
                clip_quantile=None, 
                logit=args.logit_transform,
                normal=args.normal_transform,
                include_year=False,
                stride_lr=None, padding_lr=None,
                period_norm=period_norm, # TO DO: check, still valid?
                remove_climate_change_trend=args.remove_climate_change_trend if mode == "inference" else False,
                test_params=test_params,
                return_dataset=True)
            
            orog_data = full_dataset.orog.to(device)  # shape (128, 128)
            samples = []
            model.eval()
            model.to(device)
            
            start = time.time()
            
            for idx, data_batch in enumerate(test_loader_in):
                if not args.temporal:
                    x, xc, y = data_batch
                    x, xc, y = x.to(device), xc.to(device), y.to(device)
                else:
                    x_prev, xc_prev, y_prev, x, xc, y = data_batch
                    x_prev, xc_prev, y_prev, x, xc, y = x_prev.to(device), xc_prev.to(device), y_prev.to(device), x.to(device), xc.to(device), y.to(device)
                
                
                with torch.no_grad():
                    cls_ids = None
                    if args.temporal:
                        start_xc = coarse_model_marginal.sample(x_prev[:1, ...], sample_size=1).to(device)
                        # start_xc = model.coarse_model.sample(x_prev[:1, ...], sample_size=1).to(device)
                        gen = model.sample_temporal(x, sample_size=5, start_xc=start_xc[0, :, 0], x_onehot=x, cls_ids=cls_ids).to(device)
                        gen = gen.view(x.shape[0], len(args.variables), -1, 5)
                    else:
                        if args.add_orography:
                            gen = model.sample(x.to(device), sample_size=5, x_onehot=x, cls_ids=cls_ids, orog=orog_data.expand(x.shape[0], -1)).to(device)
                            gen = gen.view(x.shape[0], len(args.variables), -1, 5)
                        else:
                            gen = model.sample(x.to(device), sample_size=5, x_onehot=x, cls_ids=cls_ids).to(device)
                            gen = gen.view(x.shape[0], len(args.variables), -1, 5)
                    
                samples.append(gen.detach().cpu())
                
            end = time.time()
            print("Time taken for sampling: ", end - start)
            
            samples_norm = torch.cat(samples)
            
            start = time.time()
            # do normalisation here with larger batch size 
            batch_size_unnorm = 20000 
            n_batches = np.ceil(samples_norm.shape[0] / batch_size_unnorm)
            print(n_batches)
            samples_raw = []
            samples_counterfact_raw = []
            for i in range(int(n_batches)):
                gen = samples_norm[i * batch_size_unnorm: (i+1) * batch_size_unnorm]
                gen_raw_allvars_list = []
                
                for i in range(len(args.variables)):
                    gen_raw_var_list = []
                    
                    for j in range(gen.shape[-1]):
                        gen_raw = unnormalise(gen[:, i, :, j], 
                                              mode=mode_unnorm, 
                                              data_type=args.variables[i], 
                                              sqrt_transform=args.sqrt_transform_out, 
                                              norm_method=args.norm_method_output, 
                                              norm_stats=norm_stats[args.variables[i]], 
                                              logit=args.logit_transform, 
                                              normal=args.normal_transform,
                                              domain=args.domain,
                                              training_experiment=args.training_experiment,
                                              alpha=args.alpha_subtract_linear,
                                              test_params=config if mode == "inference" else None,
                                              mode_data="inference" if mode == "inference" else "train",
                                              gcm_name=gcm_name,
                                              period_training=period_training,)
                        
    
                        gen_raw_var_list.append(gen_raw)
                        
                    gen_raw_var = torch.stack(gen_raw_var_list, dim=-1)
                    gen_raw_allvars_list.append(gen_raw_var)
                        
                gen_raw_allvars = torch.stack(gen_raw_allvars_list, dim=1)
                
                # samples.append(gen_raw_allvars)
                samples_raw.append(gen_raw_allvars)
                    
            samples_raw = torch.cat(samples_raw, dim=0)
                        
            end = time.time()
            print("Time taken for unnormalisation: ", end - start)
                        
            # 5. Dynamic Saving
            if mode == "inference":
                save_name = f"samples_{config['period']}_{config['framework']}_{config['gcm']}.pt"
            else:
                save_name = f"samples_train.pt"
                
            save_path = os.path.join(save_dir_samples, save_name)
            torch.save(samples_raw, save_path)
