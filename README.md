# Score-Based Image-to-Image Brownian Bridge
![](res/structure.png)

This is the official implementation for the paper Score-Based Image-to-Image Brownian Bridge.

* For framework usage, see [Usage](#usage).
* For training and evaluating examples, see [Example Scripts Usage](#example-scripts-usage).
* For pre-trained models and checkpoints, see [Pre-trained Models and Checkpoints](#pre-trained-models-and-checkpoints).

## Pre-requisites
* Python >= 3.9
* [PyTorch](https://pytorch.org) >= 2.0.1
* [torchmanager-diffusion](https://github.com/kisonho/diffusion/) >= 1.0

## Usage
The SDE-BBDM model can be trained in image space or latent space. The following examples show how to train the model in image space and latent space with `SDEBBDMManager` class.

### Train SDE-BBDM in Image Space
1. Initialize a UNet model with `networks.build_unet` function, create an optimizer and a loss function. The coeffient `c_lambda` can be set optionally. The `time_steps` is the number of diffusion steps.
```python
import torch
from sde_bbdm import networks
from torchmanager import losses

# load model
model = networks.build_unet(3, 3)
c_lambda: float = ...
time_steps: int = ...

# load optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = losses.MAE()
```

2. Compile SDE-BBDM in Image Space
To train a Score-Based BBDM model in image space, compile `SDEBBDMManager` with the UNet model, optimizer, loss function.

```python
from sde_bbdm import SDEBBDMManager as Manager

# compile manager
time_steps: int = ...
manager = Manager(model, time_steps, optimizer=optimizer, loss_fn=loss_fn)
```

3. Initialize dataset and callbacks
```python
from torchmanager import callbacks, data

# load dataset
dataset: data.Dataset = ...

# initialize callbacks
callback_list: list[callbacks.Callback] = ...
```

4. Train the model
```python
# train the model
epochs: int = ...
trained_model = manager.fit(dataset, epochs=epochs, callbacks=callback_list)
```

### Train SDE-BBDM in Latent Space
To train a Score-Based BBDM model in latent space, compile `SDEBBDMManager` with pre-trained encoder and decoder loaded as `torch.nn.Module`.

```python
encoder: torch.nn.Module = ...
decoder: torch.nn.Module = ...

manager = Manager(model, time_steps, optimizer=optimizer, loss_fn=loss_fn, c_lambda=c_lambda, encoder=encoder, decoder=decoder)
```

### Evaluating the model
1. Create metrics
Initialize metrics with `metrics.Metric` class and set the metrics to the manager.

```python
from torchmanager import metrics

metric_fns: dict[str, metrics.Metric] = ...
manager.metrics = metric_fns
```

2. Run evaluation
To evaluate the model, use `test` method by set `sampling_images` as `True`. The method will sample images and return a dictionary containing the evaluation results.

```python
model.test(dataset, sampling_images=True)
```

### Evaluating the model using fast sampling
To evaluate the model using fast sampling, run `test` method by set `sampling_images` as `True` and set sampling steps as a list of integers with `fast_sampling` as `True`. The method will return a dictionary containing the evaluation results.

```python
sampling_steps: list[int] = ...
model.test(dataset, sampling_images=True, fast_sampling=True, sampling_steps=sampling_steps)
```

## Example Scripts Usage
This section describes how to use the example scripts to train and evaluate Score-Based Image-to-Image Brownian Bridge.

### Dependencies for example scripts

All the required packages for example scripts can be installed with the following command:
```bash
pip install -r requirements.txt
```

### Install package
To run examples, install the package first. The following command installs the package in editable mode.

```bash
pip install -e .
```

### Training Script
Use `train.py` to train a Score-Based Image-to-Image Brownian Bridge model. The script supports training in image space and latent space. The following examples show how to train the model in image space and latent space using edge2shoes dataset.

```bash
# go to the exmamples folder
cd examples

# train in image space
python train.py \
    edge2shoes \
    <data_dir> \
    <output_model_path>

# train in latent space
python train.py \
    edge2shoes \
    <data_dir> \
    <output_model_path> \
    -vq <vqgan_model_path>
```

Use `--show_verbose` to display the training progress bar for each epoch. Set `--device` as `cuda:<gpu_id>` to use specific GPU. Use `-use_multi_gpus` without `--device` argument to use multiple GPUs.

### Evaluation Script
Use `eval.py` and `eval_miou.py` to evaluate a Score-Based Image-to-Image Brownian Bridge model. The script supports evaluation of torchmanager checkpoints and pre-trained PyTorch model. The following examples show how to evaluate a torchmanager checkpoint or a pre-trained PyTorch model with edge2shoes dataset using fast sampling method.

```bash
# go to the exmamples folder
cd examples

# evaluate torchmanager checkpoint
python eval.py \
    edge2shoes \
    <data_dir> \
    <checkpoint_path> \
    --fast_sampling

# evaluate pre-trained PyTorch model
python eval.py \
    edge2shoes \
    <data_dir> \
    <model_path> \
    --fast_sampling \
    --t 1000 \\
    -vq <vqgan_model_path>
```

Use `eval_miou.py` to evaluate the model with mIoU metric for cityscapes. The following examples show how to evaluate a torchmanager checkpoint or a pre-trained PyTorch model using fast sampling method.

```bash
# go to the exmamples folder
cd examples

# evaluate torchmanager checkpoint
python eval_miou.py \
    <deeplabv3_model_path> \
    <data_dir> \
    <checkpoint_path> \
    --fast_sampling

# evaluate pre-trained PyTorch model
python eval_miou.py \
    <deeplabv3_model_path> \
    <data_dir> \
    <model_path> \
    --fast_sampling \
    --t 1000 \\
    -vq <vqgan_model_path>
```

Again, use `--show_verbose` to display the training progress bar for each epoch. Set `--device` as `cuda:<gpu_id>` to use specific GPU. Use `-use_multi_gpus` without `--device` argument to use multiple GPUs.

### Generation Script
Use `generate.py` to generate images from a Score-Based Image-to-Image Brownian Bridge model. The script supports generation from checkpoints and pre-trained PyTorch model. The following examples show how to generate images from a checkpoint or a pre-trained PyTorch model with edge2shoes dataset using fast sampling method.

```bash
python generate.py \
    edge2shoes \
    <data_dir> \
    <checkpoint_path> \
    --fast_sampling
```

### Pre-trained Models and Checkpoints
We used pre-trained VQGAN separated from the [official LDM OpenImage checkpoint](https://github.com/CompVis/latent-diffusion). We first exported the state dict of the VQGAN model in the checkpoints, then use `convert_vqgan.py` script to convert the state dict to our `vqgan.VQGAN` PyTorch model for easy loading. The following command shows how to convert the state dict to a PyTorch model.

```bash
python convert_vqgan.py \
    <state_dict_path> \
    <output_model_path>
```

We used the pre-trained deeplabv3 to evaluate the mIoU on Cityscapes from [here](https://github.com/VainF/DeepLabV3Plus-Pytorch). Again, we convert the checkpoints in state dict into `deeplabv3.DeepLabV3` PyTorch model using `convert_deeplabv3.py` script for easy loading. The following command shows how to convert the state dict to a PyTorch model.

```bash
python convert_deeplabv3.py \
    <state_dict_path> \
    <output_model_path>
```

<!-- The converted VQGAN and deeplabv3 models can be downloaded from the following links:
* [vqgan]()
* [deeplabv3]() -->

<!--
We also provide some checkpoints and pre-trained models on multiple datasets for Score-Based Image-to-Image Brownian Bridge. 

### Checkpoints

### Pre-trained Models -->
