CUDA_VISIBLE_DEVICES=0 nohup python uniform_gravity_1d.py --xla=True> log_uniform.log </dev/null 2>&1 &&
CUDA_VISIBLE_DEVICES=1 nohup python inversion_gravity_1d.py --xla=True> log_inversion.log </dev/null 2>&1 &