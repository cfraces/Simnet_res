CUDA_VISIBLE_DEVICES=1 nohup python segregation_2d_uniform_perm.py --xla=True > log_experiments10.log </dev/null 2>&1 &&
CUDA_VISIBLE_DEVICES=1 nohup python segregation_2d_drop.py --xla=True > log_experiments11.log </dev/null 2>&1 &&
CUDA_VISIBLE_DEVICES=1 nohup python segregation_2d_inversion.py --xla=True > log_experiments12.log </dev/null 2>&1 &&
CUDA_VISIBLE_DEVICES=1 nohup python segregation_2d_vertical.py --xla=True > log_experiments13.log </dev/null 2>&1 &