# Pytorch 2 Lightning Examples

# Bare MNIST
* [PyTorch](bare_mnist/pytorch.py)
* [Lightning](bare_mnist/lightning.py)

# DDP MNIST
* [PyTorch](ddp_mnist/pytorch.py) | +30 lines 
* [Lightning](ddp_mnist/lightning.py) | +0 lines 

# DDP MNIST + Accumulate gradients
* [PyTorch](ddp_mnist_accumulate_gradients/pytorch.py) | +5 lines 
* [Lightning](ddp_mnist_accumulate_gradients/lightning.py) | +0 lines 

# DDP MNIST + Profiling + Accumulate gradients
* [PyTorch](ddp_profiler_mnist/pytorch.py) | +25 lines 
* [Lightning](ddp_profiler_mnist/lightning.py) | +0 lines 

# DDP MNIST + Profiling + Grid

```bash
pip install lightning-grid
cd ddp_mnist_multi_nodes/
grid run --instance_type p3.8xlarge --use_spot ddp_mnist_grid/lightning.py
```

Here is [Docs](https://docs.grid.ai/platform/about-these-features/multi-node)

* [PyTorch] Need to use a cloud solution. Code changes ...
* [Lightning](ddp_profiler_mnist/lightning.py) | +0 lines 


## Lightning contains hundreds of features working together and highly tested for speed and reproducibility. I won't re-implement them all in pure PyTorch :)

Learn more with [Lighting Docs](https://pytorch-lightning.readthedocs.io/en/stable/)
