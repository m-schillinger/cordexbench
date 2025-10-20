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