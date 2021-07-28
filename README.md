# Pytorch 2 Lightning Examples

The repository will show you the followings:
* How to convert a pure `PyTorch Convolutional Neural Network Classifier` trained on Mnist to [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning).
* How to add some features to Pure PyTorch and how Lightning makes it trivial.

![Minst Dataset](https://miro.medium.com/max/800/1*LyRlX__08q40UJohhJG9Ow.png)

## Bare MNIST
* [PyTorch](bare_mnist/pytorch.py) | 137 lines 
* [Lightning](bare_mnist/lightning.py) | 126 lines 

## Add DDP Support
* [PyTorch](ddp_mnist/pytorch.py) | 190 lines
* [Lightning](ddp_mnist/lightning.py) | 126 lines: -64 lines

## Add DDP Spawn Support
* [PyTorch](ddp_mnist_spawn/lightning.py) | 197 lines
* [Lightning](ddp_mnist_spawn/lightning.py) | 126 lines: -71 lines

## Add Accumulated gradients Support
* [PyTorch](ddp_mnist_accumulate_gradients/pytorch.py) | +199 lines 
* [Lightning](ddp_mnist_accumulate_gradients/lightning.py) | 126 lines: -73 lines

## Add Profiling Support
* [PyTorch](https://pytorch.org/) | +226 lines 
* [Lightning](ddp_profiler_mnist/lightning.py) | 126 lines: -100 lines

## Add DeepSpeed, FSDP, Multiple Loggers, Mutliple Profilers, TorchScript, Loop Customization, Fault Tolerant Training, etc ....
* [PyTorch](https://github.com/PyTorchLightning/pytorch-lightning) | :sob: + very large number of lines :scream: You `definitely` don't  want to do that :tired_face: 
* [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) | :fire: Still ~ 126 lines :rocket: Let's keep it simple. :heart_eyes:

Learn more with [Lighting Docs](https://pytorch-lightning.readthedocs.io/en/stable/).
PyTorch Lightning 1.4 is out ! Here is our [CHANGELOG](https://github.com/PyTorchLightning/pytorch-lightning/releases/tag/1.4.0).

Don't forget to :star: [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning).

# Training on [Grid.ai](https://www.grid.ai/)

[Grid.ai](https://www.grid.ai/) is the MLOps Platform from the creators of [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning). 

Learn more with [Grid.ai Docs](https://docs.grid.ai/platform/about-these-features/multi-node)

### 1. Install Lightning-Grid

```bash
pip install lightning-grid --upgrade
```

### 2. SEAMLESSLY TRAIN 100s OF MACHINE LEARNING MODELS ON THE CLOUD FROM YOUR LAPTOP


```bash
grid run --instance_type 4_M60_8gb ddp_mnist_grid/lightning.py --max_epochs 2 --gpus 4 --accelerator ddp
```

With [Grid DataStores](https://docs.grid.ai/products/global-cli-configs/cli-api/grid-datastores), low-latency, highly-scalable auto-versioned dataset.

```bash
grid datastore create --name mnist --source data
grid run --instance_type 4_M60_8gb --datastore_name mnist --datastore_mount_dir data ddp_mnist_grid/lightning.py --max_epochs 2 --gpus 4 --accelerator ddp

```

Add `--use_spot` to use preemptive machines.

[Grid.ai](https://www.grid.ai/) makes multi nodes training at scale easy :rocket: Training on 2 nodes with 4 GPUS using DDP Sharded :fire:


```bash
grid run --instance_type 4_M60_8gb --gpus 8 --datastore_name mnist --datastore_mount_dir data  ddp_mnist_grid/lightning.py --max_epochs 2 --num_nodes 2 --gpus 4 --precision 16 --accelerator ddp
```

Train [Andrej Karpathy](https://karpathy.ai) [minGPT](https://github.com/karpathy/minGPT) converted to [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) by [@williamFalcon](https://github.com/williamFalcon) and bencharmked with DeepSpeed by [@SeanNaren](https://github.com/SeanNaren)

```
git clone https://github.com/SeanNaren/minGPT.git
git checkout benchmark
grid run --instance_type g4dn.12xlarge --gpus 8 benchmark.py --n_layer 6 --n_head 16 --n_embd 2048 --gpus 4 --num_nodes 2 --precision 16 --batch_size 32 --plugins deepspeed_stage_3
```

Learn how to scale your scripts with [PyTorch Lighting + DeepSpeed](https://devblog.pytorchlightning.ai/accessible-multi-billion-parameter-model-training-with-pytorch-lightning-deepspeed-c9333ac3bb59)

Train a Lightning Flash [Video Classifier](https://github.com/PyTorchLightning/lightning-flash/blob/master/flash_examples/video_classification.py) 
[![Grid](https://img.shields.io/badge/rid_AI-run-78FF96.svg?labelColor=black&logo=data:image/svg%2bxml;base64,PHN2ZyB3aWR0aD0iNDgiIGhlaWdodD0iNDgiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHBhdGggZD0iTTEgMTR2MjBhMTQgMTQgMCAwMDE0IDE0aDlWMzYuOEgxMi42VjExaDIyLjV2N2gxMS4yVjE0QTE0IDE0IDAgMDAzMi40IDBIMTVBMTQgMTQgMCAwMDEgMTR6IiBmaWxsPSIjZmZmIi8+PHBhdGggZD0iTTM1LjIgNDhoMTEuMlYyNS41SDIzLjl2MTEuM2gxMS4zVjQ4eiIgZmlsbD0iI2ZmZiIvPjwvc3ZnPg==)](https://platform.grid.ai/#/runs?script=https://github.com/aribornstein/KineticsDemo/blob/188f1948725506914b67d3814073a7bec152ac0a/train.py&cloud=grid&instance=g4dn.xlarge&accelerators=1&disk_size=200&framework=lightning&script_args=train.py%20--gpus%201%20--max_epochs%203)


```py
import os

import flash
from flash.core.data.utils import download_data
from flash.video import VideoClassificationData, VideoClassifier

# 1. Create the DataModule
# Find more datasets at https://pytorchvideo.readthedocs.io/en/latest/data.html
download_data("https://pl-flash-data.s3.amazonaws.com/kinetics.zip", "./data")

datamodule = VideoClassificationData.from_folders(
    train_folder=os.path.join(os.getcwd(), "data/kinetics/train"),
    val_folder=os.path.join(os.getcwd(), "data/kinetics/val"),
    clip_sampler="uniform",
    clip_duration=1,
    decode_audio=False,
)

# 2. Build the task
model = VideoClassifier(backbone="x3d_xs", num_classes=datamodule.num_classes, pretrained=False)

# 3. Create the trainer and finetune the model
trainer = flash.Trainer(max_epochs=3)
trainer.finetune(model, datamodule=datamodule, strategy="freeze")

# 4. Make a prediction
predictions = model.predict(os.path.join(os.getcwd(), "data/kinetics/predict"))
print(predictions)

# 5. Save the model!
trainer.save_checkpoint("video_classification.pt")
```

### Credits

Credit to PyTorch Team for providing the [Bare Mnist example](https://github.com/pytorch/examples/blob/master/mnist/main.py).

Credit to Andrej Karpathy for providing an implementation of minGPT.
