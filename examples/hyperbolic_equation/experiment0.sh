CUDA_VISIBLE_DEVICES=0 nohup python buckley_godunov.py --activation_fn=selu --layer_size=1024 --skip_connections=True > log_experiments0.log </dev/null 2>&1 &&
CUDA_VISIBLE_DEVICES=0 nohup python buckley_godunov.py --activation_fn=poly --layer_size=64 --skip_connections=True > log_experiments0.log </dev/null 2>&1 &&
CUDA_VISIBLE_DEVICES=0 nohup python buckley_godunov.py --activation_fn=poly --layer_size=256 --skip_connections=True > log_experiments0.log </dev/null 2>&1 &&
CUDA_VISIBLE_DEVICES=0 nohup python buckley_godunov.py --activation_fn=poly --layer_size=512 --skip_connections=True > log_experiments0.log </dev/null 2>&1 &&
CUDA_VISIBLE_DEVICES=0 nohup python buckley_godunov.py --activation_fn=poly --layer_size=1024 --skip_connections=True > log_experiments0.log </dev/null 2>&1 &
