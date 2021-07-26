# Pytorch 2 Lightning Examples

# Bare MNIST
* [PyTorch](bare_mnist/pytorch.py)
* [Lightning](bare_mnist/lightning.py)

# DDP MNIST
* [PyTorch](ddp_mnist/pytorch.py) | +30 lines 
* [Lightning](ddp_mnist/lightning.py) | +0 lines 

# DDP MNIST + Accumulate gradients
* [PyTorch](ddp_mnist_accumulate_gradients/pytorch.py) | +5 lines 
* [Lightning](ddp_mnist_accumulate_gradients/lightning.py) | +0 lines 

# DDP MNIST + Profiling
* [PyTorch](ddp_profiler_mnist/pytorch.py) | +25 lines 
* [Lightning](ddp_profiler_mnist/lightning.py) | +0 lines 

# DDP MNIST + Profiling + Multi Nodes

```bash
pip install lightning-grid
grid run --instance_type g4dn.xlarge --framework lightning --gpus 2  lightning.py
```

Here is [Docs](https://docs.grid.ai/platform/about-these-features/multi-node)

* [PyTorch] Not supported.
* [Lightning](ddp_profiler_mnist/lightning.py) | +0 lines 


# Lightning contains hundreds of features working together and highly tested for speed and reproducibility. This is the results of 3+ years with 500+ contributors to get there.
