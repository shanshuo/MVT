# MVT: Multi-view Vision Transformer for 3D Object Recognition

This folder contains the PyTorch code for our BMVC 2021 paper [MVT: Multi-view Vision Transformer for 3D Object Recognition](https://arxiv.org/abs/2110.13083) by [Shuo Chen](https://shanshuo.github.io/), [Tan Yu](https://sites.google.com/site/tanyuspersonalwebsite/home), and [Ping Li](https://pltrees.github.io/).

If you use this code for a paper, please cite:


```
@inproceedings{Chen2021MVT,
  author    = {Shuo Chen and
               Tan Yu and
               Ping Li},
  title     = {{MVT:} Multi-view Vision Transformer for 3D Object Recognition},
  booktitle = {{BMVC}},
  year      = {2021},
}
```

We have developed a MLP-based architecture for view-based 3D object recognition. Check out our paper [R2-MLP: Round-Roll MLP for Multi-View 3D Object Recognition](https://arxiv.org/abs/2211.11085) and the accompanying [code repository](https://github.com/shanshuo/R2-MLP) for more information.

## Requirements
- PyTorch 1.7.0+

## Data Preparation
Download the ModelNet40 dataset (20 view setting) and extract it to the current folder:
```
wget https://data.airc.aist.go.jp/kanezaki.asako/data/modelnet40v2png_ori4.tar
tar -xvf modelnet40v2png_ori4.tar
```


## Training
[Download](https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth) the DeiT small model pretrained on ImageNet 2012 from the [Model Zoo](https://github.com/facebookresearch/deit/blob/main/README_deit.md).

Train the model on 2 V100 GPUs:
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_addr 127.0.0.2 --master_port 23918 --nproc_per_node=2 --use_env main.py --model deit_small_patch16_224 --epochs 100 --batch-size 8 --lr 0.001 --dataset M10 --view-num 20 --output_dir outputs --num_workers 4
```


The training log is available [here](logs/M10_small_view20_bs8_pretrainTRUE_lr0.001.log) for your reference.

**Note:** If you change `--view-num`, please remember to change `timm/models/vision_transformer.py` **line 316** accordingly:
```
x = x.reshape(B//20, N*20, C)
```


## Evaluation
Run the following command for evaluation:
```
CUDA_VISIBLE_DEVICES=0 python main.py --eval --model=deit_tiny_patch16_224 --resume=trained/model/path.pth --data-set=M10 --num_workers=4 --view-num=20 --batch-size=8
```

## Acknowledgments
This repo is based on [Deit](https://github.com/facebookresearch/deit) and [SOS](https://github.com/ntuyt/SOS). We thank the authors for their work.


