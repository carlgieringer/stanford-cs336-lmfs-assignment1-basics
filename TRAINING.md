# GPU training

## runpod.io

Connecting:

```sh
runpod_host=216.81.245.26
runpod_port=40061
ssh root@$runpod_host -p $runpod_port -i ~/.ssh/id_ed25519

cd /workspace/
```

## Lambda

```sh
lambda_host=192.222.56.85
ssh ubuntu@$lambda_host

cd stanford-cs336
```

## Initialization

Initialize the instance:

```sh
git clone https://github.com/carlgieringer/stanford-cs336-lmfs-assignment1-basics.git
mkdir -p stanford-cs336-lmfs-assignment1-basics/data/checkpoints
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# Unnecessary on Lambda/Ubuntu
source $HOME/.local/bin/env
cd stanford-cs336-lmfs-assignment1-basics
# Get API key from https://wandb.ai/authorize
uv run wandb login
```

### TinyStories

Copy the training data:

```sh
scp -i ~/.ssh/id_ed25519 -P $runpod_port\
 data/tokens-TinyStoriesV2-GPT4-valid.npy\
 root@${runpod_host}:/workspace/stanford-cs336-lmfs-assignment1-basics/data/tokens-TinyStoriesV2-GPT4-valid.npy
scp -i ~/.ssh/id_ed25519 -P $runpod_port\
 data/tokens-TinyStoriesV2-GPT4-train.npy\
 root@${runpod_host}:/workspace/stanford-cs336-lmfs-assignment1-basics/data/tokens-TinyStoriesV2-GPT4-train.npy
```

Kick off training:

```sh
cd /workspace/stanford-cs336-lmfs-assignment1-basics
run_name=TinyStories-single-with-validation-a100-no-compile
uv run python cs336_basics/training.py\
 --action=RunSingleTraining\
 --data-path=data/tokens-TinyStoriesV2-GPT4-train.npy\
 --validation-data-path=data/tokens-TinyStoriesV2-GPT4-valid.npy\
 --run-name=$run_name\
 --gradient-log-interval=100\
 --total-steps=10_000\
 --validation-interval=50\
 --early-stopping-patience=5\
 --early-stopping-min-delta=0.001\
 --wandb-project=stanford-cs336-language-model\
 --wandb-entity=carl-gieringer-self
```

### OWT

Copy the training data:

```sh
scp\
 data/tokens-owt_valid.npy\
 ubuntu@${lambda_host}:/home/ubuntu/stanford-cs336/stanford-cs336-lmfs-assignment1-basics/data/tokens-owt_valid.npy
scp\
 data/tokens-owt_train.npy\
 ubuntu@${lambda_host}:/home/ubuntu/stanford-cs336/stanford-cs336-lmfs-assignment1-basics/data/tokens-owt_train.npy
```

`ssh` back into the host.

```sh
# Copy data locally (faster?)
cp stanford-cs336/stanford-cs336-lmfs-assignment1-basics/data/tokens-owt_valid.npy stanford-cs336-lmfs-assignment1-basics/data/tokens-owt_valid.npy
cp stanford-cs336/stanford-cs336-lmfs-assignment1-basics/data/tokens-owt_train.npy stanford-cs336-lmfs-assignment1-basics/data/tokens-owt_train.npy

cd stanford-cs336/stanford-cs336-lmfs-assignment1-basics/

# Install GPU torch (CUDA 12.8)
uv remove torch
uv pip install torch --index-url https://download.pytorch.org/whl/cu128

uv run python cs336_basics/training.py\
 --action=RunSingleTraining\
 --data-path=data/tokens-owt_train.npy\
 --validation-data-path=data/tokens-owt_valid.npy\
 --run-name=owt-single-run\
 --compile-model\
 --learning-rate=0.001\
 --total-steps=100_000\
 --gradient-log-interval=100\
 --validation-interval=100\
 --early-stopping-patience=100\
 --early-stopping-min-delta=0.01\
 --wandb-project=stanford-cs336-language-model\
 --wandb-entity=carl-gieringer-self
 ```

### Post training

Copy the final snapshot locally to save it.

```sh
final_snapshot_filename=${run_name}-final.pt
scp -i ~/.ssh/id_ed25519 -P $runpod_port\
 root@${runpod_host}:/workspace/stanford-cs336-lmfs-assignment1-basics/data/checkpoints/${final_snapshot_filename}\
 data/checkpoints/${final_snapshot_filename}
```

TODO:

- OWT: Change context len and batch size
- tmux
- Update sweep training to keep:
  - latest snapshot for current sweep
  - snapshot with best validation loss
  - final snapshot
- Update training data loader not to be random and instead to cover entire training dataset
  (optional max length)
- Use scalene on training
