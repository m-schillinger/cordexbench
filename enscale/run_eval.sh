## # NEW

# python eval_multi_step_coarse_from_super.py --variable tasmax --domain ALPS --version tasmax-ALPS-1 --training_experiment Emulator_hist_future --norm_option none --nicolai_layers
# python eval_multi_step_coarse_from_super.py --variable pr --domain ALPS --version pr-ALPS-scalepw --training_experiment Emulator_hist_future --norm_option scalepw --nicolai_layers
# python eval_multi_step_coarse_from_super.py --variable pr --domain ALPS --version pr-ALPS-scalarin --training_experiment Emulator_hist_future --norm_option scalar_in --nicolai_layers

# python eval_multi_step_coarse_from_super.py --variable tasmax --domain ALPS --version tasmax-ALPS-normpw-past-future --training_experiment Emulator_hist_future --norm_option pw --nicolai_layers --past_future

python eval_multi_step_coarse_from_super.py --variable tasmax --domain ALPS --version tasmax-ALPS-normperperiod-past-future-standard --training_experiment Emulator_hist_future --norm_option per_period --nicolai_layers --past_future --period_norm past
python eval_multi_step_coarse_from_super.py --variable tasmax --domain ALPS --version tasmax-ALPS-normperperiod-past-future-new --training_experiment Emulator_hist_future --norm_option per_period --nicolai_layers --past_future --period_norm future

python eval_multi_step_coarse_from_super.py --variable tasmax --domain ALPS --version tasmax-ALPS-normsubtractlin-past-future-predictors-detrended --training_experiment Emulator_hist_future --norm_option subtract_linear --nicolai_layers --remove_climate_change_trend --past_future
python eval_multi_step_coarse_from_super.py --variable tasmax --domain ALPS --version tasmax-ALPS-normsubtractlin-past-future-predictors-raw --training_experiment Emulator_hist_future --norm_option subtract_linear --nicolai_layers --past_future

python eval_multi_step_coarse_from_super.py --variable tasmax --domain ALPS --version tasmax-ALPS-normsubtractlin-past-future-predictors-detrended-v2 --training_experiment Emulator_hist_future --norm_option subtract_linear --nicolai_layers --remove_climate_change_trend --past_future

python eval_multi_step_coarse_from_super.py --variable pr --domain ALPS --version pr-ALPS-normsubtractlin-past-future-predictors-detrended-v2 --training_experiment Emulator_hist_future --norm_option subtract_linear --nicolai_layers --remove_climate_change_trend --past_future
python eval_multi_step_coarse_from_super.py --variable pr --domain ALPS --version pr-ALPS-normscalepw-past-future --training_experiment Emulator_hist_future --norm_option subtract_linear --nicolai_layers --past_future

# test eval for new set of runs
python eval_multi_step_coarse_from_super.py --variable tasmax --domain ALPS --version normpw_v1 --training_experiment Emulator_hist_future --norm_option pw --nicolai_layers --mode train

# --- HOPEFULLY LAST EVAL RUNS ----

python eval_multi_step_coarse_from_super.py --variable tasmax --domain ALPS --version normpw_v2 --training_experiment Emulator_hist_future --norm_option pw --nicolai_layers --mode train
python eval_multi_step_coarse_from_super.py --variable tasmax --domain ALPS --version normpw_v2 --training_experiment Emulator_hist_future --norm_option pw --nicolai_layers --mode inference

# try subtract linear

CUDA_VISIBLE_DEVICES=2 python eval_multi_step_coarse_from_super.py \
            --variable tasmax \
            --domain NZ \
            --version linex_v1 \
            --training_experiment Emulator_hist_future \
            --norm_option subtract_linear \
            --nicolai_layers \
            --mode train \
            --alpha_subtract_linear 1000; \
CUDA_VISIBLE_DEVICES=2 python eval_multi_step_coarse_from_super.py \
            --variable tasmax \
            --domain NZ \
            --version linex_v1 \
            --training_experiment Emulator_hist_future \
            --norm_option subtract_linear \
            --nicolai_layers \
            --mode inference \
            --alpha_subtract_linear 1000