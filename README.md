# Pytorch 2 Lightning Examples

The repository shows :
* how to convert a pure PyTorch Convolutional Neural Network Classifier trained on Mnist to [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) 
* how Lightning makes it trivial to use more features.

![Minst Dataset](https://miro.medium.com/max/800/1*LyRlX__08q40UJohhJG9Ow.png)

# Bare MNIST
* [PyTorch](bare_mnist/pytorch.py)
* [Lightning](bare_mnist/lightning.py)

# DDP MNIST
* [PyTorch](ddp_mnist/pytorch.py) | +30 lines 
* [Lightning](ddp_mnist/lightning.py) | +0 lines 

# DDP MNIST + Accumulate gradients
* [PyTorch](ddp_mnist_accumulate_gradients/pytorch.py) | +35 lines 
* [Lightning](ddp_mnist_accumulate_gradients/lightning.py) | +0 lines 

# DDP MNIST + Profiling + Accumulate gradients
* [PyTorch](ddp_profiler_mnist/pytorch.py) | +60 lines 
* [Lightning](ddp_profiler_mnist/lightning.py) | +0 lines 

# DDP MNIST + Grid.ai

[Grid.ai](https://www.grid.ai/) is the MLOps Platform from the creators of PyTorch Lightning. 

```bash
pip install lightning-grid
cd ddp_mnist_grid/
grid run --instance_type p3.8xlarge --use_spot lightning.py
```

Here is [Docs](https://docs.grid.ai/platform/about-these-features/multi-node)

* [PyTorch] Need to use a cloud solution. Code changes ...
* [Lightning](ddp_profiler_mnist/lightning.py) | +0 lines 


## Lightning contains hundreds of features working together and highly tested for reproducibility, scalabitlity and inter-operability.

Learn more with [Lighting Docs](https://pytorch-lightning.readthedocs.io/en/stable/)


### Credits

Credit to PyTorch Team for providing the [Bare Mnist example](https://github.com/pytorch/examples/blob/master/mnist/main.py).
