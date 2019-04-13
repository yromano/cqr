# Conformalized Quantile Regression

Python implementation of conformalized quantile regression [1] methodology.

Conformal prediction is a technique for constructing prediction intervals that attain valid coverage in finite samples, without making distributional assumptions. Despite this appeal, existing conformal methods can be unnecessarily conservative because they form intervals of constant or weakly varying length across the input space. **Conformalized quantile regression (CQR)** [1] is a new method that is fully adaptive to heteroscedasticity and often more efficient than other conformal methods. It combines conformal prediction with classical quantile regression, inheriting the advantages of both. CQR is also supported by a theoretical guarantee of valid coverage.

[1] Yaniv Romano, Evan Patterson, and Emmanuel Candes, “Conformalized quantile regression.” 2019.

## Getting Started

This package is self-contained and implemented in python.

Part of the code is a taken from the nonconformist package available at https://github.com/donlnz/nonconformist. One may refer to the nonconformist repository to view other applications of conformal prediction.  

### Prerequisites

* python
* numpy
* scipy
* scikit-learn
* scikit-garden
* pytorch
* pandas

### Installing

The development version is available here on github:
```bash
git clone https://github.com/yromano/cqr.git
```

## Usage

Please refer to [real_data_example.ipynb](real_data_example.ipynb) for basic usage. Comparisons to competitive methods and additional usage examples of this package can be found in [synthetic_data_example.ipynb](synthetic_data_example.ipynb).

## Reproducible Research

The code available under /reproducible_experiments/ in the repository replicates the experimental results in [1].

### Publicly Available Datasets

* [Blog](https://archive.ics.uci.edu/ml/datasets/BlogFeedback): BlogFeedback data set.

* [Bio](https://archive.ics.uci.edu/ml/datasets/Physicochemical+Properties+of+Protein+Tertiary+Structure): Physicochemical  properties  of  protein  tertiary  structure  data  set.

* [Bike](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset): Bike  sharing  dataset  data  set.

* [Community](http://archive.ics.uci.edu/ml/datasets/communities+and+crime): Communities   and   crime   data   set.

* [STAR](https://www.rdocumentation.org/packages/AER/versions/1.2-6/topics/STAR): C.M. Achilles, Helen Pate Bain, Fred Bellott, Jayne Boyd-Zaharias, Jeremy Finn, John Folger, John Johnston, and Elizabeth Word. Tennessee’s Student Teacher Achievement Ratio (STAR) project, 2008.

* [Concrete](http://archive.ics.uci.edu/ml/datasets/concrete+compressive+strength): Concrete compressive strength data set.

* [Facebook Variant 1 and Variant 2](https://archive.ics.uci.edu/ml/datasets/Facebook+Comment+Volume+Dataset): Facebook  comment  volume  data  set.

### Data subject to copyright/usage rules

The Medical Expenditure Panel Survey (MPES) data can be downloaded following [this explanation](https://github.com/IBM/AIF360/blob/master/aif360/data/raw/meps/README.md) provided in [IBM's AIF360](https://github.com/IBM/AIF360) github repository.

* [MEPS_19](https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-181): Medical expenditure panel survey,  panel 19. The features and response are extracted using [meps_dataset_panel19_fy2015.py](https://github.com/IBM/AIF360/blob/master/aif360/datasets/meps_dataset_panel19_fy2015.py), **excluding** the threshold used to construct binary labels.

* [MEPS_20](https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-181): Medical expenditure panel survey,  panel 20. The features and response are extracted using [meps_dataset_panel20_fy2015.py](https://github.com/IBM/AIF360/blob/master/aif360/datasets/meps_dataset_panel20_fy2015.py), **excluding** the threshold used to construct binary labels.

* [MEPS_21](https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-192): Medical expenditure panel survey,  panel 21. The features and response are extracted using [meps_dataset_panel21_fy2016.py](https://github.com/IBM/AIF360/blob/master/aif360/datasets/meps_dataset_panel21_fy2016.py), **excluding** the threshold used to construct binary labels.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
