# CDSML
**C**lassification of **D**rug **S**ensitivity using **M**achine **L**earning applies a manifold learning method on the discretized sensitive/resistant lables and can efficiently predict the sensitive or resistant cell line-drug pairs.
>Fatemeh Ahmadi Moughari, Changiz Eslahchi; CDSML: Classification of Drug Sensitivity using Machine Learning

This repository contains the implementated codes of CDSML, off-the-shelf machine learning methods used in the paper, tuning the hyper-parametrs, and depicting the figures of paper. It also contains the preprocessed data and computed similarity matrices for cell lines and drugs in all dataset and tissue specific conditions.

## A giude to run CDSML
Please make sure to have the following libraries installed.
#### Required libraries
Python 3.6 and upper:
- Numpy
- sklearn
- Argparse
- random
- copy
- math
- sys

#### Input parameters
To execute the codes, the user must provide three input files
- `label_dirc`: the directory to a file which contains the binary sensitivity matrix (Number of rows = number of cell lines and number of columns = number of drugs)
- `simC_dirc`: the directory to a file that is a square matrix containing the similarity of cell lines
- `simD_dirc`: the directory to a file that is a square matrix containing the similarity of drugs.
- `dim`: the dimension of latent space
- `miu`: the regularization coefficient for latent matrices
- `lambda`: the coefficient that controls the similarity conservation while manifold learning
- `CV`: the number of folds in cross validation
- `repetition`: the number of repeting the cross validation 

The binary sensitivity matrix for GDSC is presented in `Data/Features/GDSC_R_sensitive.csv`. Moreover, the required similarity files are provided in `Data/Similarities` both for all dataset  and for tissue specific conditions. There are several types of cell line similarity based on Expression, Mutation, and CNV.
The recommended values for hyper-parametrs are `dim=0.7`, `miu=2.7`, `lambda=4`, `CV=5`, `repetition=30`.

__Command__

The following command is a sample of executing ADRML
```sh
python CDSML.py label_dirc=../Data/Features/B_Sensitivity.csv  simC_dirc=../Data/SC_GeneExpression.csv simD_dric=../Data/SD_Chemical.csv dim=0.7 miu=2.7 lambda=4 CV=5 repetition=30
```

## Contact

Please do not hesitate to contact us at (f.ahmadi.moughari@gmail.com) or (ch.eslahchi@sbu.ac.ir) if there is any question. 

