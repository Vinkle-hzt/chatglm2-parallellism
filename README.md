# chatglm2 parallellism test

create docker iamge by `docker/Dockerfile`

note: set `--shm-size="128G"` while build container 

install requirements.txt

use train.sh to start training

run monitor

```bash
python usage_monitor.py
```

draw performance graph
```bash
python read.py
