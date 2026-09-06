# LightSplat

**LightSplat** is a real-time RGB-D SLAM system that combines a LightGlue-based
front-end with Gaussian Splatting reconstruction and loop closure. It is designed
to keep tracking lightweight while preserving the dense, view-consistent mapping
quality of Gaussian SLAM.

<p align="center">
  <img src="assets/lightsplat.png" alt="LightSplat teaser" width="100%">
</p>

LightSplat builds on the Gaussian-SLAM pipeline and extends it with a fast visual
front-end, submap-level loop correction, and practical dataset support for indoor
RGB-D sequences.

## Setup

Clone the repository with its submodules:

```bash
git clone --recursive https://github.com/saguru23/LightSplat.git
cd LightSplat
git submodule update --init --recursive
```

Create the environment with CUDA 11.8, then install the Python dependencies:

```bash
conda create -n lightsplat python=3.10 -y
conda activate lightsplat

conda install -c nvidia/label/cuda-11.8.0 cuda=11.8 cuda-toolkit=11.8 cuda-nvcc=11.8 -y
conda install pytorch==2.1.2 torchvision==0.16.2 pytorch-cuda=11.8 faiss-gpu=1.8.0 -c pytorch -c nvidia -y

pip install -r requirements.txt
pip install -e thirdparty/Hierarchical-Localization
pip install -e thirdparty/LightGlue
```

## Dataset Download

Download TUM RGB-D:

```bash
bash scripts/download_tum.sh
```

Download Replica:

```bash
bash scripts/download_replica.sh
```

The scripts place datasets under `data/`. After downloading, edit the config file
and set the correct sequence path:

```yaml
data:
  input_path: data/TUM_RGBD-SLAM/rgbd_dataset_freiburg1_desk
  output_path: output/tum_rgbd_desk
```

For ScanNet, RealSense, or ROS input, update the corresponding config file
manually.

## Run

Run the full SLAM pipeline:

```bash
python run_slam.py configs/tum_rgbd.yaml
```

Run only the LightGlue front-end:

```bash
python -m src.tools.run_slam configs/tum_rgbd.yaml
```

## Acknowledgements

This project builds on [Gaussian-SLAM](https://github.com/VladimirYugay/Gaussian-SLAM)
and [LoopSplat](https://github.com/GradientSpaces/LoopSplat). We thank the authors
for their open-source work.

## License

This project is released under the MIT License. See `LICENSE` for details.
