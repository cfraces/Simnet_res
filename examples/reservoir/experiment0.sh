CUDA_VISIBLE_DEVICES=0 nohup python wave_2d_source.py --xla=True > log_experiments00.log </dev/null 2>&1 &&
CUDA_VISIBLE_DEVICES=0 nohup python segregation_2d_uniform_perm.py --xla=True > log_experiments00.log </dev/null 2>&1 &&
CUDA_VISIBLE_DEVICES=0 nohup python segregation_2d_drop.py --xla=True > log_experiments01.log </dev/null 2>&1 &&
CUDA_VISIBLE_DEVICES=0 nohup python segregation_2d_inversion.py --xla=True > log_experiments02.log </dev/null 2>&1 &&
CUDA_VISIBLE_DEVICES=0 nohup python segregation_2d_vertical.py --xla=True > log_experiments03.log </dev/null 2>&1 &