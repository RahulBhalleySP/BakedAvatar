<div align="center">

<h1>BakedAvatar: Baking Neural Fields for Real-Time Head Avatar Synthesis</h1>

<div>
    <a href='' target='_blank'>Hao-Bin Duan<sup>1</sup></a>&emsp;
    <a href='http://miaowang.me/' target='_blank'>Miao Wang<sup>1, 2</sup></a>&emsp;
    <a href='' target='_blank'>Jin-Chuan Shi<sup>1</sup></a>&emsp;
    <a href='' target='_blank'>Xu-Chuan Chen<sup>1</sup></a>&emsp;
    <a href='https://yanpei.me/' target='_blank'>Yan-Pei Cao<sup>3</sup></a>
</div>
<div>
    <sup>1</sup>State Key Laboratory of Virtual Reality Technology and Systems, Beihang University&emsp;
    <sup>2</sup>Zhongguancun Laboratory&emsp;
    <sup>3</sup>ARC Lab, Tencent PCG
</div>
<div>
    <a href='https://dl.acm.org/doi/10.1145/3618399'>ACM Transactions on Graphics (SIGGRAPH Asia 2023)</a>
</div>
<div>

<a target="_blank" href="https://arxiv.org/abs/2311.05521">
  <img src="https://img.shields.io/badge/arXiv-2311.05521-b31b1b.svg" alt="arXiv Paper"/>
</a>
<a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fbuaavrcg%2FBakedAvatar&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false"/></a>
</div>


<h4>TL;DR</h4>
<h5>BakedAvatar takes monocular video recordings of a person and produces a mesh-based representation for real-time 4D head avatar synthesis on various devices including mobiles.</h5>

### [Paper](https://dl.acm.org/doi/10.1145/3618399) | [Project Page](https://buaavrcg.github.io/BakedAvatar/)

<br>

<tr>
    <img src="./assets/teaser.jpg" width="100%"/>
</tr>

</div>


## Setup
First, clone this repo:
```bash
git clone https://github.com/buaavrcg/BakedAvatar
cd BakedAvatar
```

Then, install the required environment. We recommend using [Anaconda](https://www.anaconda.com/) to manage your python environment. You can setup the required environment by the following commands:
```bash
conda env create -f environment.yml
conda activate BakedAvatar
```

Or you can setup the required environment manually:
```bash
conda create -n BakedAvatar python=3.10
conda activate BakedAvatar
# Install Pytorch (or follow specific instructions for your GPU on https://pytorch.org/get-started/locally/)
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
# Install various required libraries
pip install accelerate configargparse chumpy opencv-python pymeshlab trimesh scikit-image xatlas matplotlib tensorboard tqdm torchmetrics face-alignment
# Install Pytorch3D (Or follow instructions in https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)
conda install pytorch3d -c pytorch3d  # Linux only
# Install nvdiffrast
git clone https://github.com/NVlabs/nvdiffrast
cd nvdiffrast && pip install . && cd ..
# Install mise (for levelset extraction)
pip install Cython && pip install code/utils/libmise/
```

Finally, download [FLAME model](https://flame.is.tue.mpg.de/download.php), choose FLAME 2020 and unzip it, copy 'generic_model.pkl' into ./code/flame/FLAME2020

---

## macOS Apple Silicon (M1/M2/M3/M4) Setup

> **Note:** This setup uses MPS (Metal Performance Shaders) instead of CUDA. Stage-1 (implicit field training/inference) runs fully on MPS. Stage-2/3 (mesh baking) requires nvdiffrast which currently requires CUDA and is not supported on macOS.

### Prerequisites
- Xcode with Command Line Tools: `xcode-select --install`
- [pyenv](https://github.com/pyenv/pyenv) or similar for Python management
- Homebrew

### Create virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install PyTorch (MPS-enabled)
```bash
pip install torch torchvision torchaudio
```

### Install core packages
```bash
pip install accelerate configargparse chumpy opencv-python pymeshlab trimesh scikit-image \
            xatlas matplotlib tensorboard tqdm torchmetrics face-alignment cmake gdown Cython
```

### Install pytorch3d from source (macOS requires SDK path fix)
```bash
SDK=$(xcrun --sdk macosx --show-sdk-path)
MACOSX_DEPLOYMENT_TARGET=13.0 SDKROOT=$SDK LDFLAGS="-L${SDK}/usr/lib" \
  pip install "git+https://github.com/facebookresearch/pytorch3d.git" --no-build-isolation
```

### Install libmise
```bash
SDK=$(xcrun --sdk macosx --show-sdk-path)
MACOSX_DEPLOYMENT_TARGET=13.0 SDKROOT=$SDK LDFLAGS="-L${SDK}/usr/lib" \
  pip install --no-build-isolation code/utils/libmise/
```

### Configure accelerate for MPS (single process)
```bash
cat > ~/.cache/huggingface/accelerate/default_config.yaml << 'EOF'
compute_environment: LOCAL_MACHINE
distributed_type: NO
mixed_precision: 'no'
num_machines: 1
num_processes: 1
use_cpu: false
EOF
```

### FLAME model
FLAME 2020 requires registration at https://flame.is.tue.mpg.de/download.php. Alternatively, [FLAME 2023 Open](https://flame.is.tue.mpg.de/download.php) (no registration) has a compatible structure:
```bash
mkdir -p code/flame/FLAME2020
cp /path/to/flame2023_Open.pkl code/flame/FLAME2020/generic_model.pkl
```

### Known code patches for macOS

**`code/model/metrics.py`** — auto-detect MPS/CPU for face_alignment:
```python
device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device=device)
```

**`code/dataset/real.py`** — fix float64 keypoints (MPS doesn't support float64):
```python
# line ~88: change to explicit float32
halfsize_bbox = np.array([img_res[0], img_res[1], img_res[0], img_res[1]], dtype=np.float32) / 2
```

### Run inference on macOS (Stage-1 only)
```bash
cd code
# Use --num_workers 0 to avoid macOS multiprocessing issues
# Use --img_res 256 256 for lower memory usage (512x512 may OOM)
accelerate launch scripts/runner.py -c config/subject1.yaml -t test \
  --img_res 256 256 --num_workers 0
```

---

## Kaggle Notebook Setup (GPU / Multi-GPU)

A ready-to-use Kaggle notebook is provided at [`BakedAvatar_Kaggle.ipynb`](./BakedAvatar_Kaggle.ipynb). Import it directly on Kaggle.

**Quick steps:**
1. Go to [Kaggle](https://www.kaggle.com) → **New Notebook** → **File** → **Import Notebook** → upload `BakedAvatar_Kaggle.ipynb`
2. Enable GPU accelerator: **Settings** → **Accelerator** → T4 GPU (or 2×T4 for multi-GPU)
3. Enable internet access in Settings
4. Upload FLAME model as a Kaggle dataset (see notebook instructions)
5. Run all cells

---

## Download Training Data
We use the same data format as in [IMavatar](https://github.com/zhengyuf/IMavatar) and [PointAvatar](https://github.com/zhengyuf/PointAvatar).
You can download a preprocessed dataset from [subject 1](https://dataset.ait.ethz.ch/downloads/IMavatar_data/data/subject1.zip), [subject 2](https://dataset.ait.ethz.ch/downloads/IMavatar_data/data/subject2.zip), then unzip the files into `data/datasets` folder. You should be able to see the paths of one subject's videos structured like `data/datasets/<subject_name>/<video_name>`.
To generate your own dataset, please follow the instructions in the [IMavatar repo](https://github.com/zhengyuf/IMavatar/tree/main/preprocess).

We also provide an example of the pre-trained checkpoint [here](https://drive.google.com/file/d/137TTr8GENZmPZ-Me1SatymMCiVDKqDPR/view?usp=drive_link).

```bash
# Download dataset
wget https://dataset.ait.ethz.ch/downloads/IMavatar_data/data/subject1.zip
unzip subject1.zip -d data/datasets/

# Download pretrained checkpoint (requires 7zip for extraction)
pip install gdown
gdown 137TTr8GENZmPZ-Me1SatymMCiVDKqDPR -O data/pretrained.7z
7z x data/pretrained.7z -o data/

# Set up expected directory structure (the checkpoint is for "yufeng" dataset)
mkdir -p data/experiments
mv data/yufeng/034_8layer_Rfreq12_Mfreq5_flamegtdist_eyeweight2 data/experiments/subject1
ln -sf $(pwd)/data/datasets/subject1/subject1 data/datasets/yufeng
```

## Train implicit fields (Stage-1)
```bash
# Configure your training (use DDP?), see https://huggingface.co/docs/accelerate for details
accelerate config

cd code
accelerate launch scripts/runner.py -c config/subject1.yaml -t train
```

## Bake meshes and textures (Stage-2)
```bash
# Extract the meshes
accelerate launch scripts/runner.py -c config/subject1.yaml -t mesh_export
# Precompute the textures and export MLP weights
# Note: change the path to the actual mesh_data.pkl path if you use a different config
accelerate launch scripts/runner.py -c config/subject1.yaml -t texture_export --mesh_data_path ../data/experiments/subject1/mesh_export/iter_30000/marching_cube/res_init16_up5/mesh_data.pkl
```

## Fine tuning (Stage-3)
```bash
# Fine-tune the textures with higher resolution (512x512)
accelerate launch scripts/runner.py -c config/subject1.yaml -t fine_tuning --img_res 512 512 --batch_size 1 --mesh_data_path ../data/experiments/subject1/mesh_export/iter_30000/marching_cube/res_init16_up5/mesh_data.pkl
```

## Run evaluation
```bash
# evaluate fine-tuned meshes (result of stage-3)
accelerate launch scripts/runner.py -c config/subject1.yaml -t test --img_res 512 512 --use_finetune_model --mesh_data_path ../data/experiments/subject1/finetune_mesh_data/iter_30000/mesh_data.pkl

# evaluate baked meshes (result of stage-2)
accelerate launch scripts/runner.py -c config/subject1.yaml -t test --img_res 512 512 --use_finetune_model --mesh_data_path ../data/experiments/subject1/mesh_export/iter_30000/marching_cube/res_init16_up5/mesh_data.pkl

# evaluate implicit fields (result of stage-1) (the rendering speed will be much slower)
accelerate launch scripts/runner.py -c config/subject1.yaml -t test --img_res 512 512

# run cross-identity reenactment using meshes in PyTorch code
# you may replace the reenact_data_dir with the path to the reenactment dataset and replace the reenact_subdirs with the subdirectories names
# if you would like to see reenactment results of implicit fields, remove --use_finetune_model
accelerate launch scripts/runner.py -c config/subject1.yaml -t test --img_res 512 512 \
  --use_finetune_model --mesh_data_path ../data/experiments/subject1/finetune_mesh_data/iter_30000/mesh_data.pkl \
  --reenact_data_dir ../data/datasets/soubhik --reenact_subdirs test
```

## Export assets and run the real-time web demo
```bash
# export baked meshes and textures for the web demo
python scripts/unpack_pkl.py ../data/experiments/subject1/finetune_mesh_data/iter_30000/mesh_data.pkl --output ./mesh_data

# export the FLAME parameter sequence for reenactment
# The flame_params.json are from the files in the train and test subfolders of the dataset (e.g., ../data/datasets/soubhik/train/flame_params.json)
# You may export the sequences from the same identity for self-reenactment, or from different identities for cross-identity reenactment.
python scripts/export_flame_sequence.py <path to 1st flame_params.json> <path to 2nd flame_params.json> ... --output ./sequence_data
```

Copy the exported `mesh_data` directory and `sequence_data` directory into the root of the web demo and start the server.
The dictionary structure should be like:
```
web_demo
├── jsUtils
├── mesh_data
├── sequence_data
└── src
```
Make sure that you have installed Npm and Node.js, then run the following commands:
```bash
cd web_demo
npm install
npm run build
npm install --global serve
serve
```
Then, open your browser and visit `http://localhost:8080/`. To run the real-time reenactment, you can select one of the buttons with the name of the sequence in the web demo.

## Citation
If you find our code or paper useful, please cite as:
```
@article{bakedavatar,
  author = {Duan, Hao-Bin and Wang, Miao and Shi, Jin-Chuan and Chen, Xu-Chuan and Cao, Yan-Pei},
  title = {BakedAvatar: Baking Neural Fields for Real-Time Head Avatar Synthesis},
  year = {2023},
  issue_date = {December 2023},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3618399},
  doi = {10.1145/3618399},
  volume = {42},
  number = {6},
  journal = {ACM Trans. Graph.},
  month = {sep},
  articleno = {225},
  numpages = {14}
}
```
