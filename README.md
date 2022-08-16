# Anonymization for Skeleton Action Recognition

This repository is the official implementation of **'Anonymization for Skeleton Action Recognition'**

## Anonymization framework
 <div align="center">
    <img src="resource/framework.jpg">
</div>



## Prerequisites
- Python3
- [Pytorch](https://pytorch.org/)
- Run `pip install -r requirements.txt` for installing other python libraries
- We use [Wandb](https://wandb.ai/site) for experiment tracking 

## Compile cuda extensions

  ```
  cd ./model/Temporal_shift
  bash run.sh
  ```


## Data Preparation
- We use [NTU RGB+D](https://github.com/shahroudy/NTURGB-D) skeleton-only datasets (`nturgbd_skeletons_s001_to_s017.zip`).
- After downloading datasets, generate the skeleton data with this command.
```
python data_gen/ntu_gendata.py --data_path <path to nturgbd+d_skeletons>
```

## Training 

To train the models in the paper, run this command:

```train
python main.py --config ./config/train_adver_resnet.yaml
python main.py --config ./config/train_adver_unet.yaml
```


## Pre-trained Models

We provide two pre-trained model with NTU60. You can download pretrained models here:
- [Google Drive](https://drive.google.com/drive/folders/1R7fooJbVv2an42Xdt2Phvi8I3GTB5CxG?usp=sharing)

Model|Anonymizer network|Re-iden. acc.|Action acc.
-|-|-|-
./save_models/pretrained_resnet.pt|ResNet|4.20%|91.75%
./save_models/pretrained_unet.pt|UNet|5.70%|91.45%

To test the pre-trained models given above, run this command:

```train
python main.py --config ./config/train_adver_resnet.yaml
python main.py --config ./config/train_adver_unet.yaml
```


## Acknowledgements
This code is based on [Shift-GCN](https://github.com/kchengiva/Shift-GCN). Also, we use [U-Net](https://github.com/milesial/Pytorch-UNet) for anonymizer network. Thanks to the original authors!☺️
