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

[Grid.ai](https://www.grid.ai/) makes multi nodes training at scale easy :rocket: Training on 2 nodes with 4 GPUS using DDP Sharded :fire:


```bash
grid run --instance_type 4_M60_8gb --gpus 8 ddp_mnist_grid/lightning.py --max_epochs 2 --num_nodes 2 --gpus 4 --precision 16 --accelerator ddp_sharded
```

Train [Andrej Karpathy](https://karpathy.ai) [minGPT](https://github.com/karpathy/minGPT) converted to [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) by [@williamFalcon](https://github.com/williamFalcon) and bencharmked with DeepSpeed by [@SeanNaren](https://github.com/SeanNaren)

```
git clone https://github.com/SeanNaren/minGPT.git
git checkout benchmark
grid run --instance_type g4dn.12xlarge --gpus 8 benchmark.py --n_layer 6 --n_head 16 --n_embd 2048 --gpus 4 --num_nodes 2 --precision 16 --batch_size 32 --plugins deepspeed_stage_3
```

Learn how to scale your scripts with [PyTorch Lighting + DeepSpeed](https://devblog.pytorchlightning.ai/accessible-multi-billion-parameter-model-training-with-pytorch-lightning-deepspeed-c9333ac3bb59)

### Credits

Credit to PyTorch Team for providing the [Bare Mnist example](https://github.com/pytorch/examples/blob/master/mnist/main.py).

Credit to Andrej Karpathy for providing an implementation of minGPT.
