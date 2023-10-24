gnome-terminal -- bash -c "python3 /home/ubuntu/autodl_one_layer/scripts/train/evaluate_no_available.py --add_dropout --dropout_prob 0.1 --num_mini_batch 16 --nn_type attention_resnet --episode_length 190 --ppo_epoch 5 --n_rollout_threads 16 --entropy_coef 0.01 --gamma 0.99 --batch_expand_times 2 --transpose_time 1 --hidden_size 512 --max_grad_norm 0.5 --d_model 32 --nhead 4 --lr_rate 0.00005; exec bash"
sleep 2

