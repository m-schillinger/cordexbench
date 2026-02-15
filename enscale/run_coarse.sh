# marginal coarse model
python train_only-coarse.py --num_layer 6 --hidden_dim 200 --out_act None --save_name '_normal_preproc_zerosconstant' --precip_zeros constant --method eng_2step --variables tas pr sfcWind rsds \
 --variables_lr tas pr sfcWind rsds psl --num_epochs 500 --norm_method_input normalise_scalar --norm_method_output uniform --normal_transform --kernel_size_lr 16 --layer_shrinkage 1 --noise_dim 10 \
 --agg_norm_loss mean --lambda_norm_loss_loc 1  --p_norm_loss_loc 4 --p_norm_loss_batch 4 --preproc_layer --preproc_dim 50 --save_model_every 50

# temporal coarse model
python train_only-coarse.py --num_layer 6 --hidden_dim 200 --out_act None --save_name '_normal_preproc_zerosconstant' --precip_zeros constant --method eng_temporal --variables tas pr sfcWind rsds \
--variables_lr tas pr sfcWind rsds psl --num_epochs 500 --norm_method_input normalise_scalar --norm_method_output uniform --normal_transform --kernel_size_lr 16 --layer_shrinkage 1 --noise_dim 10 \
--agg_norm_loss mean --lambda_norm_loss_loc 1  --p_norm_loss_loc 4 --p_norm_loss_batch 4 --preproc_layer --preproc_dim 50 --save_model_every 50

### CORDEXBENCH ALPS ###
CUDA_VISIBLE_DEVICES=0 python train_only-coarse.py --num_layer 6 --hidden_dim 200 --out_act None --save_name '' --method eng_2step --variables tasmax \
 --variables_lr all --num_epochs 100 --norm_method_input None --norm_method_output normalise_pw --kernel_size_lr 16 --layer_shrinkage 1 --noise_dim 10 \
 --agg_norm_loss mean --lambda_norm_loss_loc 1  --p_norm_loss_loc 4 --p_norm_loss_batch 4 --preproc_layer --preproc_dim 20 --save_model_every 50

 CUDA_VISIBLE_DEVICES=0 python train_only-coarse.py --num_layer 6 --hidden_dim 200 --out_act None --save_name '' --method eng_2step --variables tasmax --domain ALPS \
 --variables_lr all --num_epochs 100 --norm_method_input None --norm_method_output None --kernel_size_lr 16 --layer_shrinkage 1 --noise_dim 10 \
 --agg_norm_loss mean --lambda_norm_loss_loc 1  --p_norm_loss_loc 4 --p_norm_loss_batch 4 --preproc_layer --preproc_dim 20 --save_model_every 50

CUDA_VISIBLE_DEVICES=0 python train_only-coarse.py --num_layer 6 --hidden_dim 200 --out_act None --save_name '' --method eng_2step --variables tasmax --domain NZ \
 --variables_lr all --num_epochs 100 --norm_method_input None --norm_method_output None --kernel_size_lr 16 --layer_shrinkage 1 --noise_dim 10 \
 --agg_norm_loss mean --lambda_norm_loss_loc 1  --p_norm_loss_loc 4 --p_norm_loss_batch 4 --preproc_layer --preproc_dim 20 --save_model_every 50

# deterministic baseline
CUDA_VISIBLE_DEVICES=0 python train_only-coarse.py --num_layer 6 --hidden_dim 200 --out_act None --save_name '' --method nn_det --variables tasmax \
 --variables_lr all --num_epochs 500 --norm_method_input None --norm_method_output normalise_pw --kernel_size_lr 16 --layer_shrinkage 1 --noise_dim 10 \
 --agg_norm_loss mean --lambda_norm_loss_loc 1  --p_norm_loss_loc 4 --p_norm_loss_batch 4 --preproc_layer --preproc_dim 20 --save_model_every 50

 # deterministic baseline, no data normalization
CUDA_VISIBLE_DEVICES=0 python train_only-coarse.py --num_layer 6 --hidden_dim 200 --out_act None --save_name '' --method nn_det --variables tasmax \
 --variables_lr all --num_epochs 100 --norm_method_input None --norm_method_output None --kernel_size_lr 16 --layer_shrinkage 1 --noise_dim 10 \
 --agg_norm_loss mean --lambda_norm_loss_loc 1  --p_norm_loss_loc 4 --p_norm_loss_batch 4 --preproc_layer --preproc_dim 20 --save_model_every 50

 # test without preproc layer
 CUDA_VISIBLE_DEVICES=0 python train_only-coarse.py --num_layer 6 --hidden_dim 200 --out_act None --save_name '_no-preproc' --method nn_det --variables tasmax \
 --variables_lr all --num_epochs 500 --norm_method_input None --norm_method_output normalise_pw --kernel_size_lr 16 --layer_shrinkage 1 --noise_dim 10 \
 --agg_norm_loss mean --lambda_norm_loss_loc 1  --p_norm_loss_loc 4 --p_norm_loss_batch 4 --save_model_every 50

 # deterministic baseline NZ, no data normalization
CUDA_VISIBLE_DEVICES=0 python train_only-coarse.py --num_layer 6 --hidden_dim 200 --out_act None --save_name '' --method nn_det --variables tasmax --domain NZ \
 --variables_lr all --num_epochs 100 --norm_method_input None --norm_method_output None --kernel_size_lr 16 --layer_shrinkage 1 --noise_dim 10 \
 --agg_norm_loss mean --lambda_norm_loss_loc 1  --p_norm_loss_loc 4 --p_norm_loss_batch 4 --preproc_layer --preproc_dim 20 --save_model_every 50


  CUDA_VISIBLE_DEVICES=0 python train_only-coarse.py --num_layer 6 --hidden_dim 200 --out_act None --save_name '_v2' --method eng_2step --variables tasmax --domain ALPS \
 --variables_lr all --num_epochs 500 --norm_method_input None --norm_method_output None --kernel_size_lr 16 --layer_shrinkage 1 --noise_dim 10 \
 --agg_norm_loss mean --lambda_norm_loss_loc 1  --p_norm_loss_loc 4 --p_norm_loss_batch 4 --preproc_layer --preproc_dim 20 --save_model_every 50

 # TRY PRECIP
   CUDA_VISIBLE_DEVICES=1 python train_only-coarse.py --num_layer 6 --hidden_dim 200 --out_act None --save_name '' --method eng_2step --variables pr --domain ALPS \
 --variables_lr all --num_epochs 500 --norm_method_input None --norm_method_output None --kernel_size_lr 16 --layer_shrinkage 1 --noise_dim 10 \
 --agg_norm_loss mean --lambda_norm_loss_loc 1  --p_norm_loss_loc 4 --p_norm_loss_batch 4 --preproc_layer --preproc_dim 20 --save_model_every 50 \
 --sqrt_transform_out --sqrt_transform_in


   CUDA_VISIBLE_DEVICES=1 python train_only-coarse.py --num_layer 6 --hidden_dim 200 --out_act None --save_name '_norm-in-normalise_scalar' --method eng_2step --variables pr --domain ALPS \
 --variables_lr all --num_epochs 500 --norm_method_input normalise_scalar --norm_method_output None --kernel_size_lr 16 --layer_shrinkage 1 --noise_dim 10 \
 --agg_norm_loss mean --lambda_norm_loss_loc 1  --p_norm_loss_loc 4 --p_norm_loss_batch 4 --preproc_layer --preproc_dim 20 --save_model_every 50 \
 --sqrt_transform_out --sqrt_transform_in

    CUDA_VISIBLE_DEVICES=0 python train_only-coarse.py --num_layer 6 --hidden_dim 200 --out_act None --save_name '_norm-in-normalise_scalar' --method eng_2step --variables tasmax --domain ALPS \
 --variables_lr all --num_epochs 2000 --norm_method_input normalise_scalar --norm_method_output None --kernel_size_lr 16 --layer_shrinkage 1 --noise_dim 10 \
 --agg_norm_loss mean --lambda_norm_loss_loc 1  --p_norm_loss_loc 4 --p_norm_loss_batch 4 --preproc_layer --preproc_dim 20 --save_model_every 50

     CUDA_VISIBLE_DEVICES=1 python train_only-coarse.py --num_layer 6 --hidden_dim 200 --out_act None --save_name '_norm-in-normalise_scalar' --method eng_2step --variables tasmax --domain ALPS \
 --variables_lr all --num_epochs 2000 --norm_method_input normalise_scalar --norm_method_output normalise_pw --kernel_size_lr 16 --layer_shrinkage 1 --noise_dim 10 \
 --agg_norm_loss mean --lambda_norm_loss_loc 1  --p_norm_loss_loc 4 --p_norm_loss_batch 4 --preproc_layer --preproc_dim 20 --save_model_every 50


     CUDA_VISIBLE_DEVICES=1 python train_only-coarse.py --num_layer 6 --hidden_dim 200 --out_act relu --save_name '_relu_norm-in-normalise_scalar' --method eng_2step --variables pr --domain ALPS \
 --variables_lr all --num_epochs 1000 --norm_method_input normalise_scalar --norm_method_output scale_pw --kernel_size_lr 16 --layer_shrinkage 1 --noise_dim 10 \
 --agg_norm_loss mean --lambda_norm_loss_loc 1  --p_norm_loss_loc 4 --p_norm_loss_batch 4 --preproc_layer --preproc_dim 20 --save_model_every 50 \
 --sqrt_transform_out --sqrt_transform_in --out_act relu

      CUDA_VISIBLE_DEVICES=0 python train_only-coarse.py --num_layer 6 --hidden_dim 200 --out_act relu --save_name '_relu_norm-in-normalise_scalar' --method eng_2step --variables pr --domain ALPS \
 --variables_lr all --num_epochs 1000 --norm_method_input normalise_scalar --norm_method_output None --kernel_size_lr 16 --layer_shrinkage 1 --noise_dim 10 \
 --agg_norm_loss mean --lambda_norm_loss_loc 1  --p_norm_loss_loc 4 --p_norm_loss_batch 4 --preproc_layer --preproc_dim 20 --save_model_every 50 \
 --sqrt_transform_out --sqrt_transform_in

#### SMALLER NETWORKS

CUDA_VISIBLE_DEVICES=0 python train_only-coarse.py --num_layer 6 --hidden_dim 100 --out_act None --save_name '_norm-in-normalise_scalar' --method eng_2step --variables tasmax --domain ALPS \
 --variables_lr all --num_epochs 2000 --norm_method_input normalise_scalar --norm_method_output normalise_pw --kernel_size_lr 16 --layer_shrinkage 1 --noise_dim 10 \
 --agg_norm_loss mean --lambda_norm_loss_loc 1  --p_norm_loss_loc 4 --p_norm_loss_batch 4 --preproc_layer --preproc_dim 20 --save_model_every 100

 CUDA_VISIBLE_DEVICES=1 python train_only-coarse.py --num_layer 6 --hidden_dim 100 --out_act None --save_name '_norm-in-normalise_scalar_nd-100' --method eng_2step --variables tasmax --domain ALPS \
 --variables_lr all --num_epochs 2000 --norm_method_input normalise_scalar --norm_method_output normalise_pw --kernel_size_lr 16 --layer_shrinkage 1 --noise_dim 100 \
 --agg_norm_loss mean --lambda_norm_loss_loc 1  --p_norm_loss_loc 4 --p_norm_loss_batch 4 --preproc_layer --preproc_dim 20 --save_model_every 100

  CUDA_VISIBLE_DEVICES=1 python train_only-coarse.py --num_layer 6 --hidden_dim 100 --out_act None --save_name '_norm-in-normalise_scalar_nd-100_small' --method eng_2step --variables tasmax --domain ALPS \
 --variables_lr all --num_epochs 2000 --norm_method_input normalise_scalar --norm_method_output normalise_pw --kernel_size_lr 16 --layer_shrinkage 1 --noise_dim 100 \
 --agg_norm_loss mean --lambda_norm_loss_loc 1  --p_norm_loss_loc 4 --p_norm_loss_batch 4 --preproc_layer --preproc_dim 10 --save_model_every 100

# smaller network also for precip

  CUDA_VISIBLE_DEVICES=1 python train_only-coarse.py --num_layer 6 --hidden_dim 100 --out_act relu --save_name '_relu_norm-in-normalise_scalar_nd-100' --method eng_2step --variables pr --domain ALPS \
 --variables_lr all --num_epochs 2000 --norm_method_input normalise_scalar --norm_method_output scale_pw --kernel_size_lr 16 --layer_shrinkage 1 --noise_dim 100 \
 --agg_norm_loss mean --lambda_norm_loss_loc 1  --p_norm_loss_loc 4 --p_norm_loss_batch 4 --preproc_layer --preproc_dim 20 --save_model_every 100 \
  --sqrt_transform_out --sqrt_transform_in

### CORDEXBENCH SA ###

 CUDA_VISIBLE_DEVICES=0 python train_only-coarse.py --num_layer 6 --hidden_dim 100 --out_act None --save_name '_norm-in-normalise_scalar_nd-100' --method eng_2step --variables tasmax --domain SA \
 --variables_lr all --num_epochs 2000 --norm_method_input normalise_scalar --norm_method_output normalise_pw --kernel_size_lr 16 --layer_shrinkage 1 --noise_dim 100 \
 --agg_norm_loss mean --lambda_norm_loss_loc 1  --p_norm_loss_loc 4 --p_norm_loss_batch 4 --preproc_layer --preproc_dim 20 --save_model_every 100

  CUDA_VISIBLE_DEVICES=0 python train_only-coarse.py --num_layer 6 --hidden_dim 200 --out_act None --save_name '_norm-in-normalise_scalar' --method eng_2step --variables tasmax --domain SA \
 --variables_lr all --num_epochs 2000 --norm_method_input normalise_scalar --norm_method_output normalise_pw --kernel_size_lr 16 --layer_shrinkage 1 --noise_dim 10 \
 --agg_norm_loss mean --lambda_norm_loss_loc 1  --p_norm_loss_loc 4 --p_norm_loss_batch 4 --preproc_layer --preproc_dim 20 --save_model_every 100

  CUDA_VISIBLE_DEVICES=1 python train_only-coarse.py --num_layer 6 --hidden_dim 100 --out_act None --save_name '_norm-in-normalise_scalar_nd-100_small' --method eng_2step --variables tasmax --domain SA \
 --variables_lr all --num_epochs 2000 --norm_method_input normalise_scalar --norm_method_output normalise_pw --kernel_size_lr 16 --layer_shrinkage 1 --noise_dim 100 \
 --agg_norm_loss mean --lambda_norm_loss_loc 1  --p_norm_loss_loc 4 --p_norm_loss_batch 4 --preproc_layer --preproc_dim 10 --save_model_every 100

#### NZ ######

# precip 
  CUDA_VISIBLE_DEVICES=1 python train_only-coarse.py --num_layer 6 --hidden_dim 100 --out_act relu --save_name '_relu_norm-in-normalise_scalar_nd-100' --method eng_2step --variables pr --domain SA \
 --variables_lr all --num_epochs 2000 --norm_method_input normalise_scalar --norm_method_output scale_pw --kernel_size_lr 16 --layer_shrinkage 1 --noise_dim 100 \
 --agg_norm_loss mean --lambda_norm_loss_loc 1  --p_norm_loss_loc 4 --p_norm_loss_batch 4 --preproc_layer --preproc_dim 20 --save_model_every 100 \
  --sqrt_transform_out --sqrt_transform_in

### other train test split

 CUDA_VISIBLE_DEVICES=1 python train_only-coarse.py --num_layer 6 --hidden_dim 20 --out_act None --save_name '_norm-in-normalise_scalar_nd-100_PAST-FUTURE-SPLIT' --method eng_2step --variables tasmax --domain ALPS \
 --variables_lr all --num_epochs 1000 --norm_method_input normalise_scalar --norm_method_output normalise_pw --kernel_size_lr 16 --layer_shrinkage 1 --noise_dim 100 \
 --agg_norm_loss mean --lambda_norm_loss_loc 1  --p_norm_loss_loc 4 --p_norm_loss_batch 4 --preproc_layer --preproc_dim 20 --save_model_every 100 --tr_te_split past_future

 CUDA_VISIBLE_DEVICES=0 python train_only-coarse.py --num_layer 6 --hidden_dim 20 --out_act None --save_name '_norm-in-normalise_scalar_nd-100_PAST-FUTURE-SPLIT' --method eng_2step --variables tasmax --domain SA \
 --variables_lr all --num_epochs 1000 --norm_method_input normalise_scalar --norm_method_output normalise_pw --kernel_size_lr 16 --layer_shrinkage 1 --noise_dim 100 \
 --agg_norm_loss mean --lambda_norm_loss_loc 1  --p_norm_loss_loc 4 --p_norm_loss_batch 4 --preproc_layer --preproc_dim 20 --save_model_every 100 --tr_te_split past_future


 CUDA_VISIBLE_DEVICES=0 python train_only-coarse.py --num_layer 6 --hidden_dim 100 --bottleneck_dim 10 --out_act None --save_name '_bd-10_norm-in-normalise_scalar_nd-100_PAST-FUTURE-SPLIT' \
  --method eng_2step --variables tasmax --domain ALPS \
 --variables_lr all --num_epochs 1000 --norm_method_input normalise_scalar --norm_method_output normalise_pw --kernel_size_lr 16 --layer_shrinkage 1 --noise_dim 100 \
 --agg_norm_loss mean --lambda_norm_loss_loc 1  --p_norm_loss_loc 4 --p_norm_loss_batch 4 --preproc_layer --preproc_dim 20 --save_model_every 100 --tr_te_split past_future

 CUDA_VISIBLE_DEVICES=1 python train_only-coarse.py --num_layer 6 --hidden_dim 100 --bottleneck_dim 10 --out_act None --save_name '_bd-10_norm-in-normalise_scalar_nd-100_PAST-FUTURE-SPLIT' \
  --method eng_2step --variables tasmax --domain SA \
 --variables_lr all --num_epochs 1000 --norm_method_input normalise_scalar --norm_method_output normalise_pw --kernel_size_lr 16 --layer_shrinkage 1 --noise_dim 100 \
 --agg_norm_loss mean --lambda_norm_loss_loc 1  --p_norm_loss_loc 4 --p_norm_loss_batch 4 --preproc_layer --preproc_dim 20 --save_model_every 100 --tr_te_split past_future

### 

CUDA_VISIBLE_DEVICES=1 python train_only-coarse.py --num_layer 6 --hidden_dim 100 --out_act None --save_name '_norm-in-normalise_scalar_nd-100_PAST-FUTURE-SPLIT' --method eng_2step --variables pr --domain ALPS \
 --variables_lr all --num_epochs 1000 --norm_method_input normalise_scalar --norm_method_output normalise_pw --kernel_size_lr 16 --layer_shrinkage 1 --noise_dim 100 \
 --agg_norm_loss mean --lambda_norm_loss_loc 1  --p_norm_loss_loc 4 --p_norm_loss_batch 4 --preproc_layer --preproc_dim 20 --save_model_every 100 --tr_te_split past_future \
 --sqrt_transform_in --sqrt_transform_out

### test new normalisation idea

CUDA_VISIBLE_DEVICES=1 python train_only-coarse.py --num_layer 6 --hidden_dim 100 --out_act None --save_name '_norm-in-normalise_per_period_nd-100_PAST-FUTURE-SPLIT' --method eng_2step --variables tasmax --domain ALPS \
 --variables_lr all --num_epochs 1000 --norm_method_input normalise_per_period --norm_method_output normalise_per_period --kernel_size_lr 16 --layer_shrinkage 1 --noise_dim 100 \
 --agg_norm_loss mean --lambda_norm_loss_loc 1  --p_norm_loss_loc 4 --p_norm_loss_batch 4 --preproc_layer --preproc_dim 20 --save_model_every 100 --tr_te_split past_future 

# train on both past and future but normalised with past
CUDA_VISIBLE_DEVICES=1 python train_only-coarse.py --num_layer 6 --hidden_dim 100 --out_act None --save_name '_norm-in-normalise_per_period_nd-100' --method eng_2step --variables tasmax --domain ALPS \
 --variables_lr all --num_epochs 1000 --norm_method_input normalise_per_period --norm_method_output normalise_per_period --kernel_size_lr 16 --layer_shrinkage 1 --noise_dim 100 \
 --agg_norm_loss mean --lambda_norm_loss_loc 1  --p_norm_loss_loc 4 --p_norm_loss_batch 4 --preproc_layer --preproc_dim 20 --save_model_every 100 

# train on both past and future; subtract linear model predictions; LR with simple normalise scalar
CUDA_VISIBLE_DEVICES=1 python train_only-coarse.py --num_layer 6 --hidden_dim 100 --out_act None --save_name '_norm-in-normalise_scalar_nd-100' --method eng_2step --variables tasmax --domain ALPS \
 --variables_lr all --num_epochs 1000 --norm_method_input normalise_scalar --norm_method_output subtract_linear --kernel_size_lr 16 --layer_shrinkage 1 --noise_dim 100 \
 --agg_norm_loss mean --lambda_norm_loss_loc 1  --p_norm_loss_loc 4 --p_norm_loss_batch 4 --preproc_layer --preproc_dim 20 --save_model_every 100 

CUDA_VISIBLE_DEVICES=0 python train_only-coarse.py --num_layer 6 --hidden_dim 100 --out_act None --save_name '_norm-in-normalise_scalar_nd-100_PAST-FUTURE-SPLIT' --method eng_2step --variables tasmax --domain ALPS \
 --variables_lr all --num_epochs 1000 --norm_method_input normalise_scalar --norm_method_output subtract_linear --kernel_size_lr 16 --layer_shrinkage 1 --noise_dim 100 \
 --agg_norm_loss mean --lambda_norm_loss_loc 1  --p_norm_loss_loc 4 --p_norm_loss_batch 4 --preproc_layer --preproc_dim 20 --save_model_every 100 --tr_te_split past_future

CUDA_VISIBLE_DEVICES=0 python train_only-coarse.py --num_layer 6 --hidden_dim 100 --out_act None --save_name '_norm-in-normalise_scalar-detrended_nd-100_PAST-FUTURE-SPLIT' --method eng_2step --variables tasmax --domain ALPS \
 --variables_lr all --num_epochs 1000 --norm_method_input normalise_scalar --norm_method_output subtract_linear --kernel_size_lr 16 --layer_shrinkage 1 --noise_dim 100 \
 --agg_norm_loss mean --lambda_norm_loss_loc 1  --p_norm_loss_loc 4 --p_norm_loss_batch 4 --preproc_layer --preproc_dim 20 --save_model_every 100 --tr_te_split past_future --remove_climate_change_trend

### precip, baseline

 CUDA_VISIBLE_DEVICES=0 python train_only-coarse.py --num_layer 6 --hidden_dim 100 --out_act None --save_name '_relu_norm-in-normalise_scalar_nd-100_PAST-FUTURE-SPLIT' --method eng_2step --variables pr --domain ALPS \
 --variables_lr all --num_epochs 1000 --norm_method_input normalise_scalar --norm_method_output scale_pw --out_act relu --kernel_size_lr 16 --layer_shrinkage 1 --noise_dim 100 \
 --agg_norm_loss mean --lambda_norm_loss_loc 1  --p_norm_loss_loc 4 --p_norm_loss_batch 4 --preproc_layer --preproc_dim 20 --save_model_every 100 --tr_te_split past_future \
 --sqrt_transform_in --sqrt_transform_out

## test ESD pseudo reality
  CUDA_VISIBLE_DEVICES=2 python train_only-coarse.py --num_layer 6 --hidden_dim 100 --out_act relu --save_name '_relu_norm-in-normalise_scalar_nd-100' --method eng_2step --variables pr --domain ALPS \
  --variables_lr all --num_epochs 1000 --norm_method_input normalise_scalar --norm_method_output scale_pw --kernel_size_lr 16 --layer_shrinkage 1 --noise_dim 100 \
  --agg_norm_loss mean --lambda_norm_loss_loc 1  --p_norm_loss_loc 4 --p_norm_loss_batch 4 --preproc_layer --preproc_dim 20 --save_model_every 100 \
  --sqrt_transform_out --sqrt_transform_in --training_experiment ESD_pseudo_reality