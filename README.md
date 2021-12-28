# GCN-HPP
Graph Convolution Network for Housing Price Prediction

## Datasets
Datasets were downloaded from kaggle.

[california](https://www.kaggle.com/camnugent/california-housing-prices)

[melbourne](https://www.kaggle.com/anthonypino/melbourne-housing-market)

## Environment setup
1. Prepare virtual environment using venv or conda if you need.
2. Install requirements using pip. If you have nvidia GPU then use requirements-cuda.txt or just use requirements.txt.

## Usage

The example usage is:

``` bash
python src/main.py \
	--device cuda \
	--model gcn \
	--hidden_dim 16 \
	--dataset california \
	--distance_limit 3 \
	--self_weight 2500 \
	--epochs 50000 \
	--early_stopping 5000
```

This is full parameter list of this code.

| **Parameter** | **Description** | **Default** |
|:--- | :--- | :---: |
| `model` | Which model to use (`mlp` or `gcn`) | `gcn` |
| `dir` | Directory where dataset is located | `data`|
|`dataset`| Dataset to train (`california` or `melbourne`) | `california`|
| `device` | Torch device to use (`cpu` or `cuda`) | `cpu`|
| `epochs` | Number of training epochs | 100000 |
| `lr` | Learning rate | 0.01 |
| `reg` | Weight regularization | 10 |
| `hidden_dim` | Hidden dimension of model | 32 |
| `distance_limit` | Upper bound of the distance to generate geometric graph | 2.5 |
| `train_rate` | Ratio of train data split | 0.85 |
| `val_rate` | Ratio of validation data split | 0.05 |
| `early_stopping` | Early stopping patient value | 2000 |
| `epsilon` | The value to prevent divide by zero when generate graph | 1e-3 |
| `self_weight` | The value to increase impact on node feature | 100 |

Use experience_example to fit hyper-parameter and generate document of it as below.

``` bash
PYTHONPATH='src' python experience_example/experience.py --device cuda
python experience_example/print_result.py
python experience_example/plot.py
```
