#!/bin/bash
wd=0
lambda=0
precip=constant
splitresid=True

if [ "$splitresid" = "False" ]; then
    split_flag="--not_split_residuals"
else
    split_flag=""
fi

# mse loss flag
if [[ "$lambda" != "0" ]]; then
    mse_flag="--add_mse_loss"
else
    mse_flag=""
fi
echo "Submitting job: wd=$wd lambda=$lambda precip=$precip split=$splitresid"

CUDA_VISIBLE_DEVICES=0 python -u train_only-super_multivariate.py \
            --out_act relu \
            --save_name "_relu_dec${wd}_lam-mse${lambda}_split-resid${splitresid}_PAST-FUTURE-SPLIT" \
            --lambda_mse_loss ${lambda} \
            $split_flag \
            $mse_flag \
            --weight_decay ${wd} \
            --precip_zeros ${precip} \
            --nicolai_layers \
            --num_neighbors_res 9 \
            --mlp_depth 2 \
            --noise_dim_mlp 0 \
            --hidden_dim 12 \
            --latent_dim 12 \
            --method eng_2step \
            --variables pr \
            --sqrt_transform_in \
            --sqrt_transform_out \
            --num_epochs 4000 \
            --norm_method_input None \
            --norm_method_output scale_pw \
            --kernel_size_hr 8 \
            --kernel_size_lr 16 \
            --sample_every_nepoch 200 \
            --save_model_every 200 \
            --variables_lr all \
            --batch_size 1024 \
            --domain ALPS \
            --tr_te_split past_future \
    > log_gpu0-pr-ALPS-05-pf.txt 2>&1 &
    
CUDA_VISIBLE_DEVICES=3 python -u train_only-super_multivariate.py \
            --out_act relu \
            --save_name "_relu_dec${wd}_lam-mse${lambda}_split-resid${splitresid}_PAST-FUTURE-SPLIT" \
            --lambda_mse_loss ${lambda} \
            $split_flag \
            $mse_flag \
            --weight_decay ${wd} \
            --precip_zeros ${precip} \
            --nicolai_layers \
            --num_neighbors_res 9 \
            --mlp_depth 2 \
            --noise_dim_mlp 0 \
            --hidden_dim 12 \
            --latent_dim 12 \
            --method eng_2step \
            --variables pr \
            --sqrt_transform_in \
            --sqrt_transform_out \
            --num_epochs 4000 \
            --norm_method_input None \
            --norm_method_output scale_pw \
            --kernel_size_hr 4 \
            --kernel_size_lr 8 \
            --sample_every_nepoch 200 \
            --save_model_every 200 \
            --variables_lr all \
            --batch_size 512 \
            --domain ALPS \
            --tr_te_split past_future \
    > log_gpu1-pr-ALPS-05-pf.txt 2>&1 &