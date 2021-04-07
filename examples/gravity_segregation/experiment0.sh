CUDA_VISIBLE_DEVICES=0 nohup python wave_2d_source.py --xla=True > log_experiments_wave.log </dev/null 2>&1 &&
CUDA_VISIBLE_DEVICES=0 nohup python drop_2d_source.py --xla=True > log_experiments_drop.log </dev/null 2>&1 &&
CUDA_VISIBLE_DEVICES=0 nohup python uniform_2d_source.py --xla=True > log_experiments_uniform.log </dev/null 2>&1 &&
CUDA_VISIBLE_DEVICES=0 nohup python flip_2d_source.py --xla=True > log_experiments_flip.log </dev/null 2>&1 &&
CUDA_VISIBLE_DEVICES=0 nohup python vertical_2d_source.py --xla=True > log_experiments_vertical.log </dev/null 2>&1 &