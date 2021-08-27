CUDA_VISIBLE_DEVICES=0 nohup python buckley_velocity_nn.py --start_lr=1e-4 --network_dir='./network_checkpoint/buckley_vel_var_1e_4'> log_0.log </dev/null 2>&1 &&
CUDA_VISIBLE_DEVICES=0 nohup python buckley_velocity_nn.py --start_lr=3e-4 --network_dir='./network_checkpoint/buckley_vel_var_3e_4'> log_0.log </dev/null 2>&1 &&
CUDA_VISIBLE_DEVICES=0 nohup python buckley_velocity_nn.py --start_lr=5e-4 --network_dir='./network_checkpoint/buckley_vel_var_5e_4'> log_0.log </dev/null 2>&1 &&
CUDA_VISIBLE_DEVICES=0 nohup python buckley_velocity_nn.py --start_lr=7e-4 --network_dir='./network_checkpoint/buckley_vel_var_7e_4'> log_0.log </dev/null 2>&1 &&
CUDA_VISIBLE_DEVICES=0 nohup python buckley_velocity_nn.py --start_lr=1e-3 --network_dir='./network_checkpoint/buckley_vel_var_1e_3'> log_0.log </dev/null 2>&1 &&
CUDA_VISIBLE_DEVICES=0 nohup python buckley_velocity_nn.py --start_lr=3e-3 --network_dir='./network_checkpoint/buckley_vel_var_3e_3'> log_0.log </dev/null 2>&1 &&
CUDA_VISIBLE_DEVICES=0 nohup python buckley_velocity_nn.py --start_lr=5e-3 --network_dir='./network_checkpoint/buckley_vel_var_5e_3'> log_0.log </dev/null 2>&1 &