# Conformalized Quantile Regression

Python implementation of conformalized quantile regression [1] methodology.

Conformal prediction is a technique for constructing prediction intervals that attain valid coverage in finite samples, without making distributional assumptions. Despite this appeal, existing conformal methods can be unnecessarily conservative because they form intervals of constant or weakly varying length across the input space. **Conformalized quantile regression (CQR)** [1] is a new method that is fully adaptive to heteroscedasticity and often more efficient than other conformal methods. It combines conformal prediction with classical quantile regression, inheriting the advantages of both. CQR is also supported by a theoretical guarantee of valid coverage.

[1] Yaniv Romano, Evan Patterson, and Emmanuel Candes, “Conformalized quantile regression.” 2019.

## Getting Started

This package is implemented in python and relies on the github repository available at https://github.com/donlnz/nonconformist.

### Prerequisites

* Python
* numpy
* scipy
* scikit-learn
* scikit-garden
* pytorch

### Installing

The development version is available here on github:
```bash
git clone https://github.com/yromano/cqr
```

## Usage

Please refer to [synthetic_data_example.ipynb](synthetic_data_example.ipynb) and [real_data_example.ipynb](real_data_example.ipynb) for usage.

## Reproducible Research

The code available under /reproducible_experiments/ in the repository replicates the experimental results in [1].

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
