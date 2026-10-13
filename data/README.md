# Dataset provenance

The project uses `sklearn.datasets.load_digits()` from scikit-learn 1.8.0.
It contains 1,797 samples, 64 features, and ten classes. Each sample is an 8x8
grayscale digit represented by numeric intensities from 0 to 16.

This is the 1,797-sample subset distributed with scikit-learn, not the whole
5,620-sample UCI dataset and not MNIST. No network download is needed after
scikit-learn is installed. Original raw data are not duplicated in this repository.

Sources:

- https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html
- https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits
- UCI citation: Alpaydin, E. and Kaynak, C. (1998). Optical Recognition of
  Handwritten Digits. UCI Machine Learning Repository. DOI: 10.24432/C50P49.
- The UCI source lists CC BY 4.0. Dataset attribution remains applicable independently
  of the MIT license for this project's original code.

Preprocessing: flatten the 8x8 image into 64 numeric features as already provided
by `load_digits`. Random Forest does not require standardization. There is no
imputation, feature selection or data augmentation. No fitted preprocessing is
applied before the split. Labels stay in 0..9.

Split: fixed stratified 80% development / 20% test. Three stratified CV folds are
created exclusively inside development data. Global indices for every split and
a SHA-256 of feature and target arrays are saved in `results/split_indices.json`.
