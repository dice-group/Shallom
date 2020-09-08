
# A shallow neural model for relation prediction

This repository contains the implementation of our approach (SHALLOM) along with experimental results for the reproducibility.

## Installation

First clone the repository:
```
git clone https://github.com/dice-group/Shallom.git.
```
Then obtain the required libraries:
```
conda env create -f environment.yml
source activate shallom
```
If anaconda virtual environment named as shallom not found, please execute the following command:
```
python -m ipykernel install --user --name shallom --display-name "Python (shallom)"
source activate shallom
```
The code is compatible with Python 3.6.4.

## Reproducing reported results
- To reproduce the reported results for our approach, please refer to the any desired .ipynb file.
- Run any desired .ipynb file


## How to cite

If you use SHALLOM, please cite the following publication:
```
@inproceedings{
  XXX,
  title={A shallow neural model for relation prediction},
  author={Caglar Demir and Diego Moussallem and Axel-Cyrille Ngonga Ngomo},
  booktitle={XXX},
  year={2020},
  url={XXX}
}
```