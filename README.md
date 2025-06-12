# BeamNG RL 
BeamNG Reinforcement Learning project

## Prerequsites
This project requires a running instance of Beamng.tech server

## Start Learning

```
python3 train_agent.py
```

or resume learning from the last save checkpoint:

```
python3 train_agent.py --resume
```

### View learning progress logs

start a tensorboard server to display learning progress
```
tensorboard --logdir=./logs/tensorboard_logs --bind_all
```

