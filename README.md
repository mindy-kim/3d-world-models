# Yoga Splat: Action Conditioned 4D Gaussians

Gaurav Suhas Gaonkar, Mindy Kim, Preetish Juneja, Bumjin Joo

## Installation

We follow the standard 3DGS setup scheme.

First, note that this code can only be run on CUDA supported machines.

Install basic requirements outlined in `requirements.txt`

```bash
git submodule update --init --recursive

pip install -r requirements.txt
pip install -e submodules/depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn
pip install -e submodules/moyo
```

## Data Preparation

We utilize known camera poses, transformations, and angles from a DNERF set. Put the `transforms_train.json`, `transforms_val.json`, and `transforms_test.json` into `data/dnerf`

Next, follow the [instructions here](https://smpl-x.is.tue.mpg.de/index.html) to download the mesh of SMPL-X. Add this to `data/models_locked_head`.

Then, follow the [instructions here](https://github.com/sha2nkt/moyo_toolkit?tab=readme-ov-file#downloading-the-dataset-in-amass-format) to download the MOYO yoga captures in SMPL-X format. Place the `mosh/` directory into `data/`

Finally, run `data_extract.py` to generate the DNERF-like data samples for our MOYO dataset.

## Training

We follow the [same instructions as provided by 4DGS](https://github.com/hustvl/4DGaussians/tree/master)

To train on the aforementioned `moyo` dataset in the standard action-conditioned task, we run the following:

```bash
python train2.py -s data/moyo-boat-pose -z data/moyo-side-plank --port 6017 --expname "moyo/boat-pose" --configs arguments/dnerf/moyo-boat-pose.py 
```

## Visualization

Then, to render, we run:

```bash
python render2.py --model_path "output/moyo/boat-pose/"  --skip_test --configs arguments/dnerf/moyo-boat-pose.py
```

