# chatglm2 parallellism test

create docker iamge by `docker/Dockerfile`

note: set `--shm-size="128G"` while build container 

install requirements.txt

change some running config like `include`, `train_path` and `num_stages` in train.sh

```bash
deepspeed  --include="localhost:0,1,2,3" \
 --master_port 5524 train_pp.py \
 --train_path data/d2q_0.json \
 --model_name_or_path /data/THUDM/chatglm2-6b/ \
 --per_device_train_batch_size 4 \
 --max_len 512 \
 --max_src_len 256 \
 --num_train_epochs 5 \
 --gradient_accumulation_steps 1 \
 --seed 1234 \
 --show_loss_step 20 \
 --num_stages 4 \
 --save_model_step 100 \
 --output_dir ./output-glm-pp
```

run monitor

```bash
python usage_monitor.py
```

draw performance graph
```bash
python read.py
