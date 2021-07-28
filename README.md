# Pytorch 2 Lightning Examples

The repository shows :
* how to convert a pure PyTorch Convolutional Neural Network Classifier trained on Mnist to [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) 
* how Lightning makes it trivial to use more features.

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

## Add DDP Accumulated gradients Support
* [PyTorch](ddp_mnist_accumulate_gradients/pytorch.py) | +199 lines 
* [Lightning](ddp_mnist_accumulate_gradients/lightning.py) | 126 lines: -73 lines

## Add Profiling Support
* [PyTorch](ddp_profiler_mnist/pytorch.py) | +226 lines 
* [Lightning](ddp_profiler_mnist/lightning.py) | 126 lines: -100 lines

## Training on Grid.ai

[Grid.ai](https://www.grid.ai/) is the MLOps Platform from the creators of PyTorch Lightning. 

```bash
pip install lightning-grid
cd ddp_mnist_grid/
grid run --instance_type p3.8xlarge --use_spot lightning.py
```

Here is [Docs](https://docs.grid.ai/platform/about-these-features/multi-node)

* [PyTorch] Need to use a cloud solution. Code changes ...
* [Lightning](ddp_profiler_mnist/lightning.py) | 126 lines 


## Lightning contains hundreds of features working together and highly tested for reproducibility, scalabitlity and inter-operability.

Learn more with [Lighting Docs](https://pytorch-lightning.readthedocs.io/en/stable/)


### Credits

Credit to PyTorch Team for providing the [Bare Mnist example](https://github.com/pytorch/examples/blob/master/mnist/main.py).
