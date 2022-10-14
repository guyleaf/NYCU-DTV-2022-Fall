# HW1: Classification

## Environment

* Python: 3.9.13
* Pytorch: 2.0.1
* Torchvision: 0.13.1 (only used for data augmentation)
* Pandas: 1.5.0
* Numpy: 1.23.3
* Matplotlib: 3.6.0
* Pillow: 9.2.0
* Tqdm: 4.64.1
* Typed-argument-parser: 1.7.2

## How to reproduce?
### Installation
#### Windows
#### Linux
### Training

```sh
python train.py --lr 0.001 --weight_decay 0.02 --batch_size 32 --epoch_size 100 --every_num_epochs_for_val 1 --num_workers 4
```

### Testing

```sh
python test.py --model HW1_311553007.pth --batch_size 32 --num_workers 4
```

## Data Augmentation

### Training

```python
transforms.RandomHorizontalFlip(p=0.5),
transforms.RandomRotation(
    45, interpolation=transforms.InterpolationMode.BILINEAR
),
transforms.ToTensor(),
transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)
```

### Testing

```python
transforms.ToTensor(),
transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)
```

## Hyper-parameters

* Loss function: Cross-Entropy with softmax
* Optimizer: Adam
* Learning rate: 0.001
* Momentum: 0.9
* Weight decay: 0.02
* Batch size: 32
* Seed: 1234

## Model

> Number of parameters: 5.12M (5,121,309)
>
> In network class,
> ![num_parameters](assets/num_parameters.png)
>
> The bias of all layers is false.

### Components

#### Dense Layer

| Layers           | Output Size (example) | Note                            |
| :--------------- | :-------------------- | :------------------------------ |
| BatchNorm + ReLU | 64 x 56 x 56          |                                 |
| Conv             | 128 x 56 x 56         | 1 x 1 conv, stride 1            |
| BatchNorm + ReLU | 128 x 56 x 56         |                                 |
| Conv_dw*         | 32 x 56 x 56          | 3 x 3 conv, stride 1, padding 1 |
| Dropout          | 32 x 56 x 56          | p=0.5                           |

#### Transition Layer

| Layers           | Output Size (example) | Note                                   |
| :--------------- | :-------------------- | :------------------------------------- |
| BatchNorm + ReLU | 256 x 56 x 56         |                                        |
| Conv             | 128 x 56 x 56         | 1 x 1 conv, stride 1                   |
| MaxPooling       | 128 x 28 x 28         | 3 x 3 max pooling, stride 2, padding 1 |

### VGG19-like network

> VGG19 + DenseNet + Depth-wise Separable Convolution

| Layers                | Output Size    | Note                                   |
| :-------------------- | :------------- | :------------------------------------- |
| Conv_dw*              | 64 x 112 x 112 | 7 x 7 conv, stride 2, padding 3        |
| BatchNorm + ReLU      | 64 x 112 x 112 |                                        |
| MaxPooling            | 64 x 56 x 56   | 3 x 3 max pooling, stride 2, padding 1 |
| DenseBlock (64)       | 256 x 56 x 56  | Dense Layer x 6                        |
| TransitionLayer (64)  | 128 x 28 x 28  |                                        |
| DenseBlock (128)      | 512 x 28 x 28  | Dense Layer x 12                       |
| TransitionLayer (128) | 256 x 14 x 14  |                                        |
| DenseBlock (256)      | 1024 x 14 x 14 | Dense Layer x 24                       |
| TransitionLayer (256) | 512 x 7 x 7    |                                        |
| DenseBlock (512)      | 2048 x 7 x 7   | Dense Layer x 16                       |
| BatchNorm + ReLU      | 1024 x 7 x 7   |                                        |
| MaxPooling            | 1024 x 1 x 1   | global max pooling                     |
| Linear                | 10             | classifier                             |

***dw**: depth-wise separable convolution

If you want to check the details, you can check this image, [detailed model structure](#Detailed-model-structure).

## Result

### Loss curve

![acc](assets/loss.png)

### Accuracy curve

![acc](assets/accuracy.png)

## Discussion

### Why does the DenseNet use Pre-Activation rather than Post-Activation?

DenseNet 論文相關作者發表另一篇 paper，[Memory-Efficient Implementation of DenseNets](https://arxiv.org/pdf/1707.06990.pdf)

裡面講述關於 DenseNet 實驗與實作上遇到的問題與改善方式，其中有提到 Pre-Activation 與 Post-Activation 的差別，Pre-Activation 能夠作用的關鍵在於使用 shared batch normalization，每一層皆使用相同的 batch normalization，以避免發生不同 layer 之間的正規化後的 features 偏差值過大 (layer2 shift outputs from layer1 with a positive constant, layer3 shift outputs from layer1 with a negative constant)。

作者在同一資料集之下測試兩種不同的實作，發現 Pre-Activation 的 error rate 會比 Post-Activation 還要來的低。

不過該篇 paper 中並沒有明確解釋為何前者比後者優，只有從實驗結果所觀察到的特性。

![pre_ac_vs_post_ac.png](assets/pre_ac_vs_post_ac.png)

### MaxPooling vs. AveragePooling

在使用相同的 hyper-parameters 下，除了第一層 7 x 7 Conv 固定使用 max pooling 以外，其餘 pooling layer 依照實驗做替換。

左圖是 MaxPooling，右圖是 AveragePooling，可以發現右圖收斂速度比左圖還要來的慢，需要更多的 epochs 來訓練，猜測可能是因為 AveragePooling 容易被數值大的 cell 所影響，導致平均值會有偏差，無法有效地將 feature maps 傳遞給下一層。

而 MaxPooling 相當於取最重要的 feature 來代表該點的 feature，比較 robust，不容易被奇怪數值影響。

#### Loss

<img src="assets/max_loss.png" alt="max_loss" style="zoom:80%;" /><img src="assets/pooling/avg_loss.png" alt="avg_loss" style="zoom:80%;" />

#### Accuracy

<img src="assets/pooling/max_accuracy.png" alt="max_acc" style="zoom:80%;" /><img src="assets/pooling/avg_accuracy.png" alt="avg_acc" style="zoom:80%;" />

## References

* 图像分类中max-pooling和average-pooling之间的异同: https://blog.csdn.net/u012193416/article/details/79432668
* DenseNet Architecture Explained with PyTorch Implementation from TorchVision: https://amaarora.github.io/2020/08/02/densenets.html
* VGG19: https://iq.opengenus.org/vgg19-architecture/
* Memory-Efficient Implementation of DenseNets: https://arxiv.org/pdf/1707.06990.pdf

### Detailed model structure

<div style="text-align: center;">
    <img src="assets/model.svg" width="40%">
</div>
