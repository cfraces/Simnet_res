CUDA_VISIBLE_DEVICES=1 nohup python reservoir_drop.py --xla=True> log_drop2d.log </dev/null 2>&1 &&
CUDA_VISIBLE_DEVICES=1 nohup python reservoir_uniform.py --xla=True> log_uniform2d.log </dev/null 2>&1 &&
CUDA_VISIBLE_DEVICES=1 nohup python reservoir_inversion.py --xla=True> log_inversion2d.log </dev/null 2>&1 &