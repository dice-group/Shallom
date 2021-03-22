# Shallom
### A shallow neural model for relation prediction
<p style="text-align: center;font-size:15px;"> <a href="https://arxiv.org/pdf/2101.09090.pdf"><img src="http://img.shields.io/badge/Paper-PDF-blue.svg"></a> <a href="https://www.youtube.com/watch?v=LUDpdgdvTQg"><img src="http://img.shields.io/badge/Youtube-Video-red.svg"></a></p>

Knowledge graph completion refers to predicting missing triples. Most approaches achieve this goal by predicting entities, given an entity and a relation. 
We predict missing triples via the relation prediction. To this end, we frame the relation prediction problem as a multi-label classification problem and propose a shallow neural model (SHALLOM) that accurately infers missing relations from entities. 
SHALLOM is analogous to C-BOW as both approaches predict a central token (p) given surrounding tokens ((s,o)). 
By virtue of its architecture, SHALLOM requires a maximum training time of 8 minutes on benchmark datasets including WN18RR, FB15K-237 and YAGO3-10. 
Hence, one does not need to win the [hardware lottery](https://research.google/pubs/pub49502/) to use SHALLOM for predicting missing information on knowledge graphs. 

## Pre-trained Models
- [DBpedia embeddings](https://hobbitdata.informatik.uni-leipzig.de/KGE/shallom/DBpedia/)
- [Carcinogenesis embeddings](https://hobbitdata.informatik.uni-leipzig.de/KGE/shallom/Carcinogenesis/)
- [Mutagenesis embeddings](https://hobbitdata.informatik.uni-leipzig.de/KGE/shallom/Mutagenesis/)
- [Biopax embeddings](https://hobbitdata.informatik.uni-leipzig.de/KGE/shallom/Biopax/)
- [Family embeddings](https://hobbitdata.informatik.uni-leipzig.de/KGE/shallom/Family/)

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
The code is compatible with Python 3.6.4.

## Reproducing reported results
- To reproduce the reported results for our approach, please refer to the any desired .ipynb file.
- Run any desired .ipynb file


## How to cite
If you use SHALLOM, please cite the following publication:
```
@inproceedings{demir2021shallow,
  title={A shallow neural model for relation prediction},
  author={Demir, Caglar and Moussallem, Diego and Ngomo, Axel-Cyrille Ngonga},
  booktitle={2021 IEEE 15th International Conference on Semantic Computing (ICSC)},
  pages={179--182},
  year={2021},
  organization={IEEE}
}
```
For any further questions, please contact:  ```caglar.demir@upb.de```
## License
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.