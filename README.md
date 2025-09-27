# CellApop
This is the official repository for CellApop.
# Installation
1. Create a virtual environment ``conda create -n cellapop python=3.10 -y`` and activate it ``conda activate cellapop``
2. ``git clone https://github.com/dddc111/CellApop``
3. Enter the CellApop folder cd CellApop and run ``pip install -r requirements.txt``
# Get Started
## Pre-train
### Data preprocessing
Processed the datasets with Cellpose, Mesmer, and CelloType and generated pseudo-labels via PixelVoting. 
Download the [Weight](https://pan.baidu.com/s/1MPUseSVNl3afR_Vd5yVyHA?pwd=APOP)
### train
```bash
python pre-train.py
```
## Fine-tune
### Data preprocessing
get your dataset and mask with [labelme](https://github.com/wkentaro/labelme)
### train
```bash
python fine-tune.py
```
### test
```bash
python test_model.py
```

## Acknowledgements
- We thank Meta AI for making the source code of [segment anything](https://github.com/facebookresearch/segment-anything) publicly available.
- We thank MouseLand for making the source code of [cellpose](https://github.com/MouseLand/cellpose) publicly available.
- We thank vanvalenlab for making the source code of [mesmer](https://github.com/vanvalenlab/cellSAM) publicly available.
- We thank tanlabcode for making the source code of [cellotype](https://github.com/tanlabcode/CelloType) publicly available.
