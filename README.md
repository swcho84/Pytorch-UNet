# U-Net: Semantic segmentation with PyTorch
<a href="#"><img src="https://img.shields.io/github/workflow/status/milesial/PyTorch-UNet/Publish%20Docker%20image?logo=github&style=for-the-badge" /></a>
<a href="https://hub.docker.com/r/milesial/unet"><img src="https://img.shields.io/badge/docker%20image-available-blue?logo=Docker&style=for-the-badge" /></a>
<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-v1.9.0-red.svg?logo=PyTorch&style=for-the-badge" /></a>
<a href="#"><img src="https://img.shields.io/badge/python-v3.6+-blue.svg?logo=python&style=for-the-badge" /></a>

![input and output for a random image in the test dataset](https://i.imgur.com/GD8FcB7.png)

---
---
## Added Information: Debugging Note
1. ground/no-ground 만 구분하기 위해 KariDB의 mask를 추출
	- 추출 시 ground=1, no-ground=0으로 입력
	- 1-channel gray image로 만들기
2. 추출하면서 3-channel 의 jpg 로 src, 1-channel gray image mask 만들었음
3. 이 코드는 1개의 class를 구분하려면 n_classes=3이어야 함
	- 그렇지 않으면 cuda연산 시 크기가 맞지 않는다는 에러 발생함
	- model의 in/out을 맞춰줄 때 오류 발생함
	- 기본적으로 2개의 class를 구분하고 있는 상태여서 그런 듯, 그렇지 않으면 어딘가에 n_classes=2를 더하고 있다고 판단됨
4. epoc/batch_size 를 비교적 잘 맞춰줘야함, 그렇지 않으면 CUDA OOM 발생함
5. scale=0.5, bilinear=True이어야 GPU 8GB에서도 실행됨, 그렇지 않으면 CUDA OOM 발생함
6. train.py 를 수정한 다음 predict.py 도 수정하였음
7. https://github.com/wang-xinyu/tensorrtx 의 unet 을 이용하여 tensorRT engine화하기로 하였음
8. gen_wts.py를 통해 pth를 wts로 변환해야 함
  - 기존의 checkpoint.pth 는 class 정보를 들고있지 않기 때문에 unet class를 별도로 불러와야 함
	- 이렇게 되면 engine을 만들어낼 수 있지만 이걸로 inference가 되지 않음
	- epoch이 모두 끝나고 나면 network 자체를 pth로 저장하도록 train.py 코드 변경하였음
	- gen_wts.py 를 통해 wts파일을 생성하였음, 이때, load만으로도 network의 모든 구조를 load 해야 함
9. https://github.com/wang-xinyu/tensorrtx 의 unet 은 1-channel output만 받아주는 코드였음
  - 해당 부분에 3-channel output 가능하도록 만들었음
	- 이렇게 해야 wts에서 정상적으로 engine 파일이 만들어짐
	- 이때, 전체 구조를 다 들고 있는 상태의 wts여야 원하는 형태의 engine file이 생성됨
10. https://github.com/wang-xinyu/tensorrtx 의 unet 에서 3-channel 중 맨 앞의 1개를 가져와야 함
  - 정상적으로 가져오려면 output을 process해주는 process_cls_result를 수정해야함, 이 부분 수정하였음
		- 특히, sigmoid 계산하는 부분이 이상해서 많이 수정하였음
	- BGR로 읽어오는 opencv image를 RGB로 변경하고 255.0으로 normalization 수행
	- inference를 위해 선언하는 CUDA관련 함수들을 초기화-사용-제거로 구분하여 배치하였음
	- prob이 confidence 이상인 경우 255로 셋팅하는 부분 디버깅하고 확정하였음
11. 완전히 수렴하기 이전의 pth를 활용하여 wts로 만들어야 함
  - predict 하면서 pth를 class 포함해서 저장하도록 코드 수정하였음
---
---
## Original ReadMe Inrofmation
Customized implementation of the [U-Net](https://arxiv.org/abs/1505.04597) in PyTorch for Kaggle's [Carvana Image Masking Challenge](https://www.kaggle.com/c/carvana-image-masking-challenge) from high definition images.

- [Quick start using Docker](#quick-start-using-docker)
- [Description](#description)
- [Usage](#usage)
  - [Docker](#docker)
  - [Training](#training)
  - [Prediction](#prediction)
- [Weights & Biases](#weights--biases)
- [Pretrained model](#pretrained-model)
- [Data](#data)

## Quick start using Docker

1. [Install Docker 19.03 or later:](https://docs.docker.com/get-docker/)
```bash
curl https://get.docker.com | sh && sudo systemctl --now enable docker
```
2. [Install the NVIDIA container toolkit:](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```
3. [Download and run the image:](https://hub.docker.com/repository/docker/milesial/unet)
```bash
sudo docker run --rm --shm-size=8g --ulimit memlock=-1 --gpus all -it milesial/unet
```

4. Download the data and run training:
```bash
bash scripts/download_data.sh
python train.py --amp
```

## Description
This model was trained from scratch with 5k images and scored a [Dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) of 0.988423 on over 100k test images.

It can be easily used for multiclass segmentation, portrait segmentation, medical segmentation, ...


## Usage
**Note : Use Python 3.6 or newer**

### Docker

A docker image containing the code and the dependencies is available on [DockerHub](https://hub.docker.com/repository/docker/milesial/unet).
You can download and jump in the container with ([docker >=19.03](https://docs.docker.com/get-docker/)):

```console
docker run -it --rm --shm-size=8g --ulimit memlock=-1 --gpus all milesial/unet
```


### Training

```console
> python train.py -h
usage: train.py [-h] [--epochs E] [--batch-size B] [--learning-rate LR]
                [--load LOAD] [--scale SCALE] [--validation VAL] [--amp]

Train the UNet on images and target masks

optional arguments:
  -h, --help            show this help message and exit
  --epochs E, -e E      Number of epochs
  --batch-size B, -b B  Batch size
  --learning-rate LR, -l LR
                        Learning rate
  --load LOAD, -f LOAD  Load model from a .pth file
  --scale SCALE, -s SCALE
                        Downscaling factor of the images
  --validation VAL, -v VAL
                        Percent of the data that is used as validation (0-100)
  --amp                 Use mixed precision
```

By default, the `scale` is 0.5, so if you wish to obtain better results (but use more memory), set it to 1.

Automatic mixed precision is also available with the `--amp` flag. [Mixed precision](https://arxiv.org/abs/1710.03740) allows the model to use less memory and to be faster on recent GPUs by using FP16 arithmetic. Enabling AMP is recommended.


### Prediction

After training your model and saving it to `MODEL.pth`, you can easily test the output masks on your images via the CLI.

To predict a single image and save it:

`python predict.py -i image.jpg -o output.jpg`

To predict a multiple images and show them without saving them:

`python predict.py -i image1.jpg image2.jpg --viz --no-save`

```console
> python predict.py -h
usage: predict.py [-h] [--model FILE] --input INPUT [INPUT ...] 
                  [--output INPUT [INPUT ...]] [--viz] [--no-save]
                  [--mask-threshold MASK_THRESHOLD] [--scale SCALE]

Predict masks from input images

optional arguments:
  -h, --help            show this help message and exit
  --model FILE, -m FILE
                        Specify the file in which the model is stored
  --input INPUT [INPUT ...], -i INPUT [INPUT ...]
                        Filenames of input images
  --output INPUT [INPUT ...], -o INPUT [INPUT ...]
                        Filenames of output images
  --viz, -v             Visualize the images as they are processed
  --no-save, -n         Do not save the output masks
  --mask-threshold MASK_THRESHOLD, -t MASK_THRESHOLD
                        Minimum probability value to consider a mask pixel white
  --scale SCALE, -s SCALE
                        Scale factor for the input images
```
You can specify which model file to use with `--model MODEL.pth`.

## Weights & Biases

The training progress can be visualized in real-time using [Weights & Biases](https://wandb.ai/).  Loss curves, validation curves, weights and gradient histograms, as well as predicted masks are logged to the platform.

When launching a training, a link will be printed in the console. Click on it to go to your dashboard. If you have an existing W&B account, you can link it
 by setting the `WANDB_API_KEY` environment variable.


## Pretrained model
A [pretrained model](https://github.com/milesial/Pytorch-UNet/releases/tag/v2.0) is available for the Carvana dataset. It can also be loaded from torch.hub:

```python
net = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True)
```
The training was done with a 50% scale and bilinear upsampling.

## Data
The Carvana data is available on the [Kaggle website](https://www.kaggle.com/c/carvana-image-masking-challenge/data).

You can also download it using the helper script:

```
bash scripts/download_data.sh
```

The input images and target masks should be in the `data/imgs` and `data/masks` folders respectively. For Carvana, images are RGB and masks are black and white.

You can use your own dataset as long as you make sure it is loaded properly in `utils/data_loading.py`.


---

Original paper by Olaf Ronneberger, Philipp Fischer, Thomas Brox:

[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

![network architecture](https://i.imgur.com/jeDVpqF.png)
