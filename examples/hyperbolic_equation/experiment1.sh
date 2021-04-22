CUDA_VISIBLE_DEVICES=1 nohup python buckley_1d_long.py --xla=True> log_long.log </dev/null 2>&1 &&
CUDA_VISIBLE_DEVICES=1 nohup buckley_1d_long_gru.py --xla=True> log_long_gru.log </dev/null 2>&1 &