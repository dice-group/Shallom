
# A shallow neural model for relation prediction

Knowledge graph completion refers to predicting missing triples. Most approaches achieve this goal by predicting entities, given an entity and a relation. 
We predict missing triples via the relation prediction. To this end, we frame the relation prediction problem as a multi-label classification problem and propose a shallow neural model (SHALLOM) that accurately infers missing relations from entities. 
SHALLOM is analogous to C-BOW as both approaches predict a central token (p) given surrounding tokens ((s,o)). 
By virtue of its architecture, SHALLOM requires a maximum training time of 8 minutes on benchmark datasets including WN18RR, FB15K-237 and YAGO3-10. 

Hence, one does not need to win the hardware lottery to use SHALLOM for predicting missing information on knowledge graphs.
More information can be found in ([Research paper](https://arxiv.org/abs/2101.09090) and [conference presentation](https://www.youtube.com/watch?v=LUDpdgdvTQg)).

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
@article{demir2021shallow,
  title={A shallow neural model for relation prediction},
  author={Demir, Caglar and Moussallem, Diego and Ngomo, Axel-Cyrille Ngonga},
  journal={arXiv preprint arXiv:2101.09090},
  year={2021}
}
```