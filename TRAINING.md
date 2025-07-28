# GPU training

## runpod.io

Connecting:

```sh
runpod_host=38.128.232.9
runpod_port=23397
ssh root@$runpod_host -p $runpod_port -i ~/.ssh/id_ed25519
```

Initialize the instance:

```sh
cd /workspace/
git clone https://github.com/carlgieringer/stanford-cs336-lmfs-assignment1-basics.git
mkdir -p stanford-cs336-lmfs-assignment1-basics/data/checkpoints
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
cd stanford-cs336-lmfs-assignment1-basics
uv run wandb login
```

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

Copy the final snapshot locally to save it.

```sh
final_snapshot_filename=${run_name}-final.pt
scp -i ~/.ssh/id_ed25519 -P $runpod_port\
 root@${runpod_host}:/workspace/stanford-cs336-lmfs-assignment1-basics/data/checkpoints/${final_snapshot_filename}\
 data/checkpoints/${final_snapshot_filename}
```

TODO:

- Try with uncompiled and gradient logging: works
- tmux
- Update sweep training to keep:
  - latest snapshot for current sweep
  - snapshot with best validation loss
  - final snapshot
- Update training data loader not to be random and instead to cover entire training dataset
  (optional max length)
