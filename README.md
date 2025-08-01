# DHO-3DGS: 3D Gaussian Splatting with Dynamic Hybrid Optimization

### How to setup 

Our experiments are done in Ubuntu 22.04.5 LTS, CUDA SDK 12.1, just for reference.

1. Follow the steps in https://github.com/graphdeco-inria/reduced-3dgs/tree/main to build the environment

### How to train

Execute the command of train.sh

### Evaluation
By default, the trained models use all available images in the dataset. To train them while withholding a test set for evaluation, use the ```--eval``` flag. This way, you can render training/test sets and produce error metrics as follows:

```
python render.py -m <path to trained model> # Generate renderings
python metrics.py -m <path to trained model> # Compute error metrics on renderings
```
### How to view

run

```
cd .\viewers\bin
.\SIBR_gaussianViewer_app.exe -m <PATH-to-output>
```
