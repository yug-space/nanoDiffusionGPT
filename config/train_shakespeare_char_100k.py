# 100k-iteration character-level NanoDiffusionGPT run on tiny Shakespeare.

out_dir = 'out-diffusion-shakespeare-char-100k'
eval_interval = 1000
eval_iters = 50
log_interval = 100
always_save_checkpoint = True

wandb_log = False
wandb_project = 'nanodiffusiongpt'
wandb_run_name = 'shakespeare-char-100k'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 32
block_size = 256

n_layer = 4
n_head = 4
n_embd = 256
dropout = 0.1

min_mask_prob = 0.05
max_mask_prob = 0.95

learning_rate = 1e-3
max_iters = 100000
lr_decay_iters = 100000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 500
