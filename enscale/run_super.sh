#!/bin/bash
wd=1e-3
lambda=0
precip=constant
splitresid=False

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

python -u train_only-super_multivariate.py \
            --n_models 8 \
            --out_act None \
            --save_name '_dec${wd}_lam-mse${lambda}_split-resid${splitresid}' \
            --lambda_mse_loss ${lambda} \
            $split_flag \
            $mse_flag \
            --weight_decay ${wd} \
            --precip_zeros ${precip} \
            --nicolai_layers \
            --num_neighbors_res 25 \
            --mlp_depth 2 \
            --noise_dim_mlp 0 \
            --hidden_dim 12 \
            --latent_dim 12 \
            --method eng_2step \
            --variables tasmax \
            --num_epochs 500 \
            --norm_method_input normalise_scalar \
            --norm_method_output normalise_pw \
            --kernel_size_hr 8 \
            --kernel_size_lr 16 \
            --sample_every_nepoch 50 \
            --save_model_every 50 \
            --variables_lr all \
            --batch_size 512
    
python -u train_only-super_multivariate.py \
            --n_models 8 \
            --out_act None \
            --save_name '_dec${wd}_lam-mse${lambda}_split-resid${splitresid}' \
            --lambda_mse_loss ${lambda} \
            $split_flag \
            $mse_flag \
            --weight_decay ${wd} \
            --precip_zeros ${precip} \
            --nicolai_layers \
            --num_neighbors_res 25 \
            --mlp_depth 2 \
            --noise_dim_mlp 0 \
            --hidden_dim 12 \
            --latent_dim 12 \
            --method eng_2step \
            --variables tasmax\
            --num_epochs 500 \
            --norm_method_input normalise_scalar \
            --norm_method_output normalise_pw \
            --kernel_size_hr 4 \
            --kernel_size_lr 8 \
            --sample_every_nepoch 50 \
            --save_model_every 50 \
            --variables_lr all \
            --batch_size 256 


python -train_only-super_multivariate.py \
        --n_models 8 \
        --out_act None \
        --save_name '_dec${wd}_lam-mse${lambda}_split-resid${splitresid}' \
        --lambda_mse_loss ${lambda} \
        $split_flag \
        $mse_flag \
        --weight_decay ${wd} \
        --precip_zeros ${precip} \
        --nicolai_layers \
        --num_neighbors_res 25 \
        --mlp_depth 2 \
        --noise_dim_mlp 0 \
        --hidden_dim 12 \
        --latent_dim 12 \
        --method eng_2step \
        --variables tasmax \
        --num_epochs 500 \
        --norm_method_input normalise_scalar \
        --norm_method_output normalise_pw \
        --kernel_size_hr 2 \
        --kernel_size_lr 4 \
        --sample_every_nepoch 50 \
        --save_model_every 50 \
        --variables_lr pr \
        --batch_size 128


python train_only-super_multivariate.py \
    --n_models 8 \
    --out_act None \
    --save_name '_dec${wd}_lam-mse${lambda}_split-resid${splitresid}' \
    --lambda_mse_loss ${lambda} \
    $split_flag \
    $mse_flag \
    --weight_decay ${wd} \
    --precip_zeros ${precip} \
    --nicolai_layers \
    --num_neighbors_res 25 \
    --mlp_depth 2 \
    --noise_dim_mlp 0 \
    --hidden_dim 12 \
    --latent_dim 12 \
    --method eng_2step \
    --variables tasmax \
    --num_epochs 500 \
    --norm_method_input normalise_scalar \
    --norm_method_output normalise_pw \
    --kernel_size_hr 1 \
    --kernel_size_lr 2 \
    --sample_every_nepoch 50 \
    --save_model_every 50 \
    --variables_lr pr \
    --batch_size 64 