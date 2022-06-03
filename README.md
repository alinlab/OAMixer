# ReMixer: Object-aware Mixing Layer for Vision Transformers

Official PyTorch implementation of
[**"ReMixer: Object-aware Mixing Layer for Vision Transformers"**](https://drive.google.com/file/d/1-3d-P9Yh3QW-CyTOmi6VTsBuW72wXSd1/view?usp=sharing) by
[Hyunwoo Kang*](https://github.com/hyunOO),
[Sangwoo Mo*](https://sites.google.com/view/sangwoomo),
and [Jinwoo Shin](http://alinlab.kaist.ac.kr/shin.html).

Our code is heavily built upon [DeiT](https://github.com/facebookresearch/deit) and [timm](https://github.com/rwightman/pytorch-image-models) repositories. We use the newer version of timm than DeiT to borrow the updated mixer implementations.
Our main contributions are in (a) `models` directory that defines the base masked model class and specific instantiations for ViT, MLP-Mixer, and ConvMixer, and (b) `transforms` directory that defines the paired transformations of image and corresponding patch labels (e.g., BigBiGAN, ReLabel).


## Installation

Install required libraries.
```
pip install -r requirements.txt
```

## Create patch labels

Create [BigBiGAN](https://github.com/anvoynov/BigGANsAreWatching) patch labels.
You can download the pretrained U-Net weights (e.g., trained on ImageNet) from the original repository.
Then, place the pretrained weights in `patch_models/pretrained`.
```
python generate_mask.py --data-set [DATASET] --output_dir [OUTPUT_PATH]
```

Create [ReLabel](https://github.com/naver-ai/relabel_imagenet) patch labels.
```
python3 generate_label.py [DATASET_PATH] [OUTPUT_PATH] --model dm_nfnet_f6 --pretrained --img-size 576 -b 32 --crop-pct 1.0
```

## Training

Train a baseline model.
```
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
 --model deit_t --batch-size 64 --data-set imagenet --output_dir [OUTPUT_PATH]
```

Apply ReMixer to the baseline model.
```
[BASE_CODE_ABOVE] --mask-attention --patch-label relabel
```

Apply [TokenLabeling](https://github.com/zihangJiang/TokenLabeling) (for both baseline model and ReMixer).
```
[BASE_CODE_ABOVE] --token-label
```

## Inference

```
python main.py --eval --model deit_t --data-set imagenet --resume [OUTPUT_PATH]/checkpoint.pth 
```

