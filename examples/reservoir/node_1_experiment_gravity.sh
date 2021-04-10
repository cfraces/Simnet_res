CUDA_VISIBLE_DEVICES=1 nohup python reservoir_uniform.py --xla=True> log_drop.log </dev/null 2>&1 &&
CUDA_VISIBLE_DEVICES=1 nohup python reservoir_uniform.py --xla=True> log_uniform.log </dev/null 2>&1 &&
CUDA_VISIBLE_DEVICES=1 nohup python reservoir_inversion.py --xla=True> log_inversion.log </dev/null 2>&1 &