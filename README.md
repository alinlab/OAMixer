# ReMixer: Object-aware Mixing Layer for Vision Transformers (CVPRW 2022)

ReMixer: Object-aware Mixing Layer for Vision Transformers (CVPRW 2022) [https://drive.google.com/file/d/1E6rXtj5h6tXiJR8Ae8u1vQcwyNyTZSVc/view?usp=sharing](https://drive.google.com/file/d/1E6rXtj5h6tXiJR8Ae8u1vQcwyNyTZSVc/view?usp=sharing)

# Training

## Baseline DeiT-T on ImageNet

```
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model deit_t --batch-size 64 --data-set imagenet --output_dir ckpt/imagenet_deit-t
```

## DeiT-T with ReMixer on ImageNet

```
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model deit_t --batch-size 64 --data-set imagenet --output_dir ckpt/imagenet_deit-t_remixer --patch-label relabel
```
