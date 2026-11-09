# :dove: PIGEON: VLM-Driven Object Navigation via Points of Interest Selection

[![arXiv](https://img.shields.io/badge/arXiv-2511.13207-b31b1b.svg)](https://arxiv.org/abs/2511.13207)
[![CoRL 2026](https://img.shields.io/badge/CoRL-2026-blue.svg)](https://2026.corl.org/)

This repository contains the official implementation of PIGEON: VLM-Driven Object Navigation via Points of Interest Selection.

PIGEON uses a vision-language model to guide object navigation by selecting points of interest from visual observations. It connects high-level semantic reasoning with 3D mapping and low-level navigation, enabling an agent to explore unfamiliar environments and navigate toward a specified object category. The framework also supports multi-floor navigation through staircase detection and traversal.

Release Status:
- Closed-set object navigation (HM3D v1, HM3D v2, and MP3D) ✅️
- Open-vocabulary object navigation (HM3D-OVON) ⬜
- Embodied question answering (A-EQA) ⬜

## :dart: Download the Dataset

### HM3D Scene Dataset
Please download the validation split of the HM3Dv2 scene data from https://github.com/matterport/habitat-matterport-3dresearch, including the following files:
```
hm3d-val-glb-v0.2.tar
hm3d-val-habitat-v0.2.tar
hm3d-val-semantic-annots-v0.2.tar
hm3d-val-semantic-configs-v0.2.tar
```
Then extract them as follows:
```shell
mkdir -p ./data/scene_datasets/hm3d/val
cd data
tar xvf hm3d-val-glb-v0.2.tar -C scene_datasets/hm3d/val
tar xvf hm3d-val-habitat-v0.2.tar -C scene_datasets/hm3d/val
tar xvf hm3d-val-semantic-annots-v0.2.tar -C scene_datasets/hm3d/val
tar xvf hm3d-val-semantic-configs-v0.2.tar -C scene_datasets/hm3d
ln -s {path to data/scene_datasets/hm3d} scene_datasets/hm3d_v0.2
```
You can refer to the file structure below:
```
data/scene_datasets/hm3d/val
├── 00800-TEEsavR23oF
├── 00801-HaxA7YrQdEC
├── ...
```

### HM3D Object Navigation Dataset
You can find object navigation task dataset in https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md. Please download  `objectnav_hm3d_v1.zip` and `objectnav_hm3d_v2.zip`, and extract then as follow:
```shell
cd data
mkdir -p datasets/objectnav/hm3d
unzip objectnav_hm3d_v1.zip -d datasets/objectnav/hm3d
unzip objectnav_hm3d_v2.zip -d datasets/objectnav/hm3d
cd datasets/objectnav/hm3d 
mv objectnav_hm3d_v1 v1
mv objectnav_hm3d_v2 v2
```

For evaluation of the `objectnav_hm3d_v2` navigation dataset, you should create a symbolic link `data/scene_datasets/hm3d_v0.2` pointing to `data/scene_datasets/hm3d/val`.

### MP3D Dataset
Please download the validation split of the MP3D scene data from https://niessner.github.io/Matterport.
You can find object navigation task dataset in https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md, similar to HM3D object navigation datasets.

## :hammer_and_wrench: Configure the Environment

### CUDA Environment
We've tested CUDA 11.8 on various versions of Nvidia GPUs. After the installation, please execute the script below for CUDA compilation later:
```bash
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
```


### Conda Environment
Below is the recommended pipeline for Conda setup:
```bash
conda create -n pigeon python=3.9 -y
conda activate pigeon 
conda install habitat-sim=0.2.5 -c conda-forge -c aihabitat
conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install cuml-cu11==23.10.0 --extra-index-url=https://pypi.nvidia.com  # ==23.10.0 for GPUs like 1070. Must be with pyarrow==12.0.0
pip install -e ".[habitat]"  #"" for zsh
pip install datasets
```

### Third-Party Dependencies
Change the present working directory to the project root directory, then execute the following instructions:
```bash
git clone https://github.com/WongKinYiu/yolov7.git
mkdir -p data
```
After that, download the checkpoints and copy them into `data`:
>https://github.com/ChaoningZhang/MobileSAM/raw/refs/heads/master/weights/mobile_sam.pt
>https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt
>https://github.com/bdaiinstitute/vlfm/raw/refs/heads/main/data/pointnav_weights.pth

## :arrow_forward: Evaluate within Habitat

### Preparation

Upgrade your openai version, and configure the api key:
```bash
pip install --upgrade openai
pip install httpx[socks]
pip install wandb
export VLM_MODEL=YOUR_VLM_MODEL
#For Local Deployment
export VLM_BASE_URL=YOUR_BASE_URL
export VLM_API_KEY=YOUR_API_KEY
#For OpenAI
export OPENAI_API_BASE=YOUR_API_BASE
export OPENAI_API_KEY=YOUR_API_KEY
#For Azure
export AZURE_OPENAI_API_KEY=YOUR_API_KEY
export AZURE_OPENAI_ENDPOINT=YOUR_END_POINT
export OPENAI_API_VERSION=YOUR_API_VERSION
```

### Run in Simulator
For HM3D and MP3D datasets:
```bash
python -m pigeon.run --num_workers 8 --habitat_cfg pigeon/configs/habitat.objectnav.hm3d.v2.yaml --num_epochs 1 --num_episodes 1000 --output_dir hm3d_v2 --visible_gpus 0 --resume
python -m pigeon.run --num_workers 8 --habitat_cfg pigeon/configs/habitat.objectnav.hm3d.v1.yaml --num_epochs 1 --num_episodes 2000 --output_dir hm3d_v1 --visible_gpus 0 --resume
python -m pigeon.run --num_workers 8 --habitat_cfg pigeon/configs/habitat.objectnav.mp3d.yaml --num_epochs 1 --num_episodes 2195 --output_dir mp3d --visible_gpus 0 --resume
```

Result Summarization:
```bash
python calculate_measure.py outputs/DATASET_OUTPUT/eval_results.csv 2195
```

Append the corresponding flag to an evaluation command when needed:

- `--video_debug`: save an MP4 visualization with debug information of each episode.
- `--episode_trace`: save per-episode JSONL logs for detailed analysis.
