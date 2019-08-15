# Uncertainty Estimation

This package contains a Python implementation of conformalized quantile regression [1] methodology. It also implements the equalized coverage framework [2] that offers a group-conditional coverage guarantee.

# Conformalized Quantile Regression

Conformal prediction is a technique for constructing prediction intervals that attain valid coverage in finite samples, without making distributional assumptions. Despite this appeal, existing conformal methods can be unnecessarily conservative because they form intervals of constant or weakly varying length across the input space. **Conformalized quantile regression (CQR)** [1] is a new method that is fully adaptive to heteroscedasticity and often more efficient than other conformal methods. It combines conformal prediction with classical quantile regression, inheriting the advantages of both. CQR is also supported by a theoretical guarantee of valid coverage.

[1] Yaniv Romano, Evan Patterson, and Emmanuel J. Candes, [“Conformalized quantile regression.”](https://arxiv.org/abs/1905.03222) 2019.

# Equalized Coverage

An important factor to guarantee a fair use of data-driven recommendation systems is that we should be able to communicate their uncertainty to decision makers. This can be accomplished by constructing prediction intervals, which provide an intuitive measure of the limits of predictive performance. To support equitable treatment, we force the construction of such intervals to be unbiased in the sense that their coverage must be equal across all protected groups of interest. We present an operational methodology that achieves this goal by offering rigorous distribution-free coverage guarantees holding in finite samples. Our methodology, **equalized coverage**, is flexible as it can be viewed as a wrapper around any predictive algorithm. We test the applicability of the proposed framework on real data, demonstrating that equalized coverage constructs unbiased prediction intervals, unlike competitive methods.

[2] Y. Romano, R. F. Barber, C. Sabbatti and E. J. Candès, [“With malice towards none: Assessing uncertainty via equalized coverage.”](https://statweb.stanford.edu/~candes/papers/EqualizedCoverage.pdf) 2019.

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

### CQR

Please refer to [cqr_real_data_example.ipynb](cqr_real_data_example.ipynb) for basic usage. Comparisons to competitive methods and additional usage examples of this package can be found in [cqr_synthetic_data_example.ipynb](cqr_synthetic_data_example.ipynb).

### Equalized Coverage

The notebook [detect_prediction_bias_example.ipynb](prediction_bias_example.ipynb) performs simple data analysis for MEPS 21 data set and detects bias in the prediction. The notebook [equalized_coverage_example.ipynb](equalized_coverage_example.ipynb) illustrates how to run the methods proposed in [2] and construct prediction intervals with equal coverage across groups.

## Reproducible Research

The code available under /reproducible_experiments/ in the repository replicates the experimental results in [1] and [2].

### Publicly Available Datasets

* [Blog](https://archive.ics.uci.edu/ml/datasets/BlogFeedback): BlogFeedback data set.

* [Bio](https://archive.ics.uci.edu/ml/datasets/Physicochemical+Properties+of+Protein+Tertiary+Structure): Physicochemical  properties  of  protein  tertiary  structure  data  set.

* [Bike](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset): Bike  sharing  dataset  data  set.

* [Community](http://archive.ics.uci.edu/ml/datasets/communities+and+crime): Communities   and   crime   data   set.

* [STAR](https://www.rdocumentation.org/packages/AER/versions/1.2-6/topics/STAR): C.M. Achilles, Helen Pate Bain, Fred Bellott, Jayne Boyd-Zaharias, Jeremy Finn, John Folger, John Johnston, and Elizabeth Word. Tennessee’s Student Teacher Achievement Ratio (STAR) project, 2008.

* [Concrete](http://archive.ics.uci.edu/ml/datasets/concrete+compressive+strength): Concrete compressive strength data set.

* [Facebook Variant 1 and Variant 2](https://archive.ics.uci.edu/ml/datasets/Facebook+Comment+Volume+Dataset): Facebook  comment  volume  data  set.

### Data subject to copyright/usage rules

The Medical Expenditure Panel Survey (MPES) data can be downloaded using the code in the folder /get_meps_data/ under this repository. It is based on [this explanation](https://github.com/IBM/AIF360/blob/master/aif360/data/raw/meps/README.md) provided by [IBM's AIF360](https://github.com/IBM/AIF360).

* [MEPS_19](https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-181): Medical expenditure panel survey,  panel 19.

* [MEPS_20](https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-181): Medical expenditure panel survey,  panel 20.

* [MEPS_21](https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-192): Medical expenditure panel survey,  panel 21. This data set is used both in [1] and [2].

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
