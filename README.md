# G1 Moves

> **[Dataset](https://huggingface.co/datasets/exptech/g1-moves)** (BVH, FBX, PKL, NPZ, ONNX policies) · **[Showcase](https://huggingface.co/spaces/exptech/g1-moves)** (interactive gallery) · **Code** (this repo)

Scripts and documentation for the G1 Moves motion capture dataset — 60 clips for the Unitree G1 humanoid robot, captured with MOVIN TRACIN and retargeted to 29-DOF joint trajectories with trained RL policies.

**All data files live on [HuggingFace](https://huggingface.co/datasets/exptech/g1-moves).** This repository contains only the processing scripts, standalone inference code, and documentation.


## What's Where

| Location | Contents |
|----------|----------|
| [HuggingFace Dataset](https://huggingface.co/datasets/exptech/g1-moves) | All motion data (BVH, FBX, PKL, NPZ), trained policies (PT, ONNX), training logs, metadata |
| [HuggingFace Space](https://huggingface.co/spaces/exptech/g1-moves) | Interactive showcase with video previews of every clip at each pipeline stage |
| [GitHub](https://github.com/experientialtech/g1-moves) (this repo) | `run_policy.py`, `retarget_all.py`, `generate_metadata.py`, `CONTROLLER.md`, documentation |


## Deploying Policies


### Option 1: Sim2Sim with mjlab (easiest)

Visualize any policy in MuJoCo simulation:

```bash
cd mjlab
uv sync
source .venv/bin/activate
# Play a trained policy
cd ..
./play.sh 


#Multi robot
MOdify play.sh

NUM_ENVS=${2:-1} to NUM_ENVS=${2:-4} #$ robot 
```

### Option 2: Using Docker

You can also run the demonstration easily using Docker, which encapsulates all dependencies.

```bash
# Build the Docker image
docker build -t strike_demo_web .

# Run the container with GPU support and port forwarding for Viser UI
docker run --gpus all -p 8080:8080 -it strike_demo_web
```

Once running, navigate to `http://localhost:8080` in your web browser to view the interactive Viser UI.
