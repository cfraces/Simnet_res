CUDA_VISIBLE_DEVICES=1 nohup python drop_2d_source.py --xla=True > log_experiments_drop1.log </dev/null 2>&1 &&
CUDA_VISIBLE_DEVICES=1 nohup python uniform_2d_source.py --start_lr=6e-4 --xla=True > log_experiments_uniform1.log </dev/null 2>&1 &&
CUDA_VISIBLE_DEVICES=1 nohup python flip_2d_source.py --start_lr=6e-4 --xla=True > log_experiments_flip1.log </dev/null 2>&1 &&
CUDA_VISIBLE_DEVICES=1 nohup python vertical_2d_source.py --start_lr=6e-4 --xla=True > log_experiments_vertical1.log </dev/null 2>&1 &