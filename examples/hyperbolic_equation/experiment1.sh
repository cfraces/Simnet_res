CUDA_VISIBLE_DEVICES=1 nohup python buckley_1d_het.py> log_het.log </dev/null 2>&1 &&
CUDA_VISIBLE_DEVICES=1 nohup python buckley_1d_het.py --max_steps=70000> log_param.log </dev/null 2>&1 &