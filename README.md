# difflearn

This is a python tool packages for differential network inference (DNI). 

-------------------------------------------------------------
This package mainly contains:

- Differential network inference models:
    - Pinv;
    - NetDiff;
    - BDgraph;
    - JGL;
    - JGLCV;

- Expression profiles simulation algorithms:
    - distributions:
        - Gaussian;
        - Exponential;
        - Mixed;
    - network structures:
        - random;
        - hub;
        - block;
        - scale-free;
- Visulization tools and some useful utilities.

## Requirments:
Before installation, you should:

1. install [pytorch](https://pytorch.org/) yourself according to your environment;
2. install [R language](https://www.r-project.org/) and R packages as follows:
    - JGL
        ```r
        install.packages( "JGL" )
        ```
    - BDgraph:
        ```r
        install.packages( "BDgraph" )
        ```
    - NetDiff:
        ```r
        library(devtools)
        install_git("https://gitlab.com/tt104/NetDiff.git")
        ```

Please note:
If you have several different versions of `R`, you should specify the version installed with above packages with:
```python
import os
os.environ["R_HOME"] = "your path to R"
```

## Installation
Easily run:
```
pip install difflearn
```



## Quick Start

```python
from difflearn.simulation import *
from difflearn.models import Random,Pinv,NetDiff,BDGraph,JointGraphicalLasso,JointGraphicalLassoCV
from difflearn.utils import *
from difflearn.visualization import show_matrix
import matplotlib.pyplot as plt

data_params = {
    'p': 10,
    'n': 1000,
    'sample_n': 100,
    'repeats': 1,
    'sparsity': [0.1, 0.1],
    'diff_ratio': [0.5, 0.5],
    'parallel_loops': 1,
    'net_rand_mode': 'BA',
    'diff_mode': 'hub',
    'target_type': 'float',
    'distribution': 'Gaussian',
    'usage': 'comparison',
}


data = ExpressionProfilesParallel(**data_params)

modelrandom = Random()
modelPinv = Pinv()
modelBDgraph = BDGraph()
modelNetDiff = NetDiff()
modelJGL = JointGraphicalLasso()
modelJGLCV = JointGraphicalLassoCV()
(sigma, delta, *X) = data[0]

modelrandom.fit(X)
modelPinv.fit(X)
modelBDgraph.fit(X)
modelNetDiff.fit(X)
modelJGL.fit(X)
modelJGLCV.fit(X)


fig, axs = plt.subplots(4, 2, figsize=(7,7))


show_matrix(vec2mat(delta)[0], ax=axs[0][0], title = 'Ground Truth')
axs[0][1].set_visible(False)
show_matrix(modelrandom.delta, ax=axs[1][0], title = 'Random')
show_matrix(modelPinv.delta, ax=axs[1][1], title = 'Pinv')
show_matrix(modelBDgraph.delta, ax=axs[2][0], title = 'BDgraph')
show_matrix(modelNetDiff.delta, ax=axs[2][1], title = 'NetDiff')
show_matrix(modelJGL.delta, ax=axs[3][0], title = 'JGL')
show_matrix(modelJGLCV.delta, ax=axs[3][1], title = 'JGLCV')
plt.tight_layout()
fig.set_dpi(300)
plt.show()
```

![Results](https://raw.githubusercontent.com/amssljc/difflearn/master/example_figures/output.png "Results")
