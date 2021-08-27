CUDA_VISIBLE_DEVICES=1 nohup python buckley_velocity_nn_x.py --start_lr=1e-4 --network_dir='./network_checkpoint/buckley_vel_var_1e_4x'> log_.log </dev/null 2>&1 &&
CUDA_VISIBLE_DEVICES=1 nohup python buckley_velocity_nn_x.py --start_lr=3e-4 --network_dir='./network_checkpoint/buckley_vel_var_3e_4x'> log_.log </dev/null 2>&1 &&
CUDA_VISIBLE_DEVICES=1 nohup python buckley_velocity_nn_x.py --start_lr=5e-4 --network_dir='./network_checkpoint/buckley_vel_var_5e_4x'> log_.log </dev/null 2>&1 &&
CUDA_VISIBLE_DEVICES=1 nohup python buckley_velocity_nn_x.py --start_lr=7e-4 --network_dir='./network_checkpoint/buckley_vel_var_7e_4x'> log_.log </dev/null 2>&1 &&
CUDA_VISIBLE_DEVICES=1 nohup python buckley_velocity_nn_x.py --start_lr=1e-3 --network_dir='./network_checkpoint/buckley_vel_var_1e_3x'> log_.log </dev/null 2>&1 &&
CUDA_VISIBLE_DEVICES=1 nohup python buckley_velocity_nn_x.py --start_lr=3e-3 --network_dir='./network_checkpoint/buckley_vel_var_3e_3x'> log_.log </dev/null 2>&1 &&
CUDA_VISIBLE_DEVICES=1 nohup python buckley_velocity_nn_x.py --start_lr=5e-3 --network_dir='./network_checkpoint/buckley_vel_var_5e_3x'> log_.log </dev/null 2>&1 &