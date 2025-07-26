# GPU training

## runpod.io

```sh
runpod_host=141.193.30.39
runpod_port=41687
ssh root@$runpod_host -p $runpod_port -i ~/.ssh/id_ed25519
```

```sh
cd /workspace/
git clone https://github.com/carlgieringer/stanford-cs336-lmfs-assignment1-basics.git
mkdir -p stanford-cs336-lmfs-assignment1-basics/data/checkpoints
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

```sh
scp -i ~/.ssh/id_ed25519 -P $runpod_port\
 data/tokens-TinyStoriesV2-GPT4-valid.npy\
 root@${runpod_host}:/workspace/stanford-cs336-lmfs-assignment1-basics/data/tokens-TinyStoriesV2-GPT4-valid.npy
scp -i ~/.ssh/id_ed25519 -P $runpod_port\
 data/tokens-TinyStoriesV2-GPT4-train.npy\
 root@${runpod_host}:/workspace/stanford-cs336-lmfs-assignment1-basics/data/tokens-TinyStoriesV2-GPT4-train.npy
```

```sh
cd /workspace/stanford-cs336-lmfs-assignment1-basics
run_name=TinyStories-single-with-validation-a100
uv run python cs336_basics/training.py\
 --action=RunSingleTraining\
 --data-path=data/tokens-TinyStoriesV2-GPT4-train.npy\
 --validation-data-path=data/tokens-TinyStoriesV2-GPT4-valid.npy\
 --run-name=$run_name\
 --compile-model\
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
 root@${runpod_host}:/workspace/stanford-cs336-lmfs-assignment1-basics/data/checkpoints/${final_snapshot_filename}$\
 data/checkpoints/${final_snapshot_filename}
```

TODO:

- tmux
- Update sweep training to keep best final snapshot and latest snapshot for current sweep.
