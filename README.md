# Energy-Based Policies

A PyTorch implementation for energy-based policies. Implements [Ranking Noise Contrastive Estimation](https://arxiv.org/abs/2309.05803) and [Implicit Behavior Cloning](https://arxiv.org/abs/2109.00137).

## Getting Started

Clone the environment and change directories. The following uses cloning via ssh:

```bash
git clone git@github.com:mht3/ebp.git
cd ebp
```

### Environment Setup

Create a new conda environment with Python 3.12.
```bash
conda create -n energy python=3.12
```

Activate the environment.
```sh
conda activate energy
```

Install torch

<details>
<summary>PyTorch on GPU</summary>
<br>
Install a CUDA enabled PyTorch that matches your system architecture.
  
```sh
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```
</details>

<details>
<summary>PyTorch on CPU Only</summary>
<br>
Alternatively, install PyTorch on the CPU.
  
```sh
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cpu
```
</details>


Install the remaining required dependencies.
```sh
pip install -r requirements.txt
```


## Results

### 2D Coordinate Regression

|             | MSE | IBC | R-NCE |
|-------------|-----------------|-----------------|
| 10 examples ||| |
| 30 examples || ||

## Citation

If you find this code useful, consider citing it along with the papers:

```bibtex
@software{taylor2026ebp,
    author = {Taylor, Matthew},
    month = {7},
    title = {{Energy-Based Policies}},
    url = {[https://github.com/kevinzakka/ibc](https://github.com/mht3/ebp)},
    version = {0.0.1},
    year = {2026}
}
```

```bibtex
@misc{florence2021implicit,
    title = {Implicit Behavioral Cloning},
    author = {Pete Florence and Corey Lynch and Andy Zeng and Oscar Ramirez and Ayzaan Wahid and Laura Downs and Adrian Wong and Johnny Lee and Igor Mordatch and Jonathan Tompson},
    year = {2021},
    eprint = {2109.00137},
    archivePrefix = {arXiv},
    primaryClass = {cs.RO}
}
```

@misc{singh2023revisitingenergybasedmodels,
      title={Revisiting Energy Based Models as Policies: Ranking Noise Contrastive Estimation and Interpolating Energy Models}, 
      author={Sumeet Singh and Stephen Tu and Vikas Sindhwani},
      year={2023},
      eprint={2309.05803},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2309.05803}, 
}

