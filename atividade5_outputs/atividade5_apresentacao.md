# Atividade 5 - Results

## Result 1 - Table

Each row below shows the nested stratified cross-validation estimate for GaussianNB.
The values use mean, standard deviation, and 95% confidence interval from the outer folds.

| Dataset variant | Classifier | Best params on full data | F1 | Accuracy | Recall | Precision |
| --- | --- | --- | --- | --- | --- | --- |
| All features | GaussianNB | {"nb__var_smoothing": 1e-06} | 0.5822 (sd=0.0068; ci95=0.0042) | 0.6874 (sd=0.0078; ci95=0.0048) | 0.9100 (sd=0.0056; ci95=0.0034) | 0.4281 (sd=0.0068; ci95=0.0042) |
| PCA | GaussianNB | {"nb__var_smoothing": 1e-09, "pca__n_components": 20} | 0.5933 (sd=0.0138; ci95=0.0086) | 0.8158 (sd=0.0054; ci95=0.0033) | 0.5613 (sd=0.0163; ci95=0.0101) | 0.6292 (sd=0.0119; ci95=0.0074) |
| SelectKBest | GaussianNB | {"nb__var_smoothing": 1e-09, "selector__k": 20} | 0.6282 (sd=0.0074; ci95=0.0046) | 0.7666 (sd=0.0062; ci95=0.0039) | 0.8233 (sd=0.0084; ci95=0.0052) | 0.5078 (sd=0.0081; ci95=0.0050) |
| SelectKBest + RandomUnderSampler | GaussianNB | {"nb__var_smoothing": 1e-09, "selector__k": 20} | 0.6324 (sd=0.0075; ci95=0.0047) | 0.7659 (sd=0.0064; ci95=0.0040) | 0.8412 (sd=0.0088; ci95=0.0055) | 0.5067 (sd=0.0082; ci95=0.0051) |

## Result 2 - Bar charts

The project now has one chart for each performance metric.
For each metric there are two versions of the chart:
- `std`: error bars with standard deviation across outer folds.
- `ci95`: error bars with 95% confidence interval across outer folds.

Use the `ci95` files if the presentation asks for confidence intervals.
Use the `std` files if the presentation asks for standard deviation.

Suggested legend text for the slide:

> Bars show the mean performance of GaussianNB for each dataset variant. Error bars show either the standard deviation or the 95% confidence interval across the outer folds of nested stratified cross-validation.

Generated chart files:
- `bar_accuracy_std.png`
- `bar_accuracy_ci95.png`
- `bar_f1_std.png`
- `bar_f1_ci95.png`
- `bar_recall_std.png`
- `bar_recall_ci95.png`
- `bar_precision_std.png`
- `bar_precision_ci95.png`

Suggested labels for the four dataset variants:
- `All features`: feature engineering + imputation + encoding + scaling, without dimensionality reduction.
- `PCA`: same preprocessing plus PCA inside the training folds.
- `SelectKBest`: same preprocessing plus supervised feature selection inside the training folds.
- `SelectKBest + RandomUnderSampler`: same preprocessing plus feature selection and class balancing inside the training folds.

Method note for the presentation:
- Performance estimation used nested stratified cross-validation.
- Hyperparameter tuning used the inner cross-validation loop with F1-score as the optimization criterion.
- Performance reporting used the outer cross-validation loop with F1, accuracy, recall, and precision.
- Imputation, encoding, scaling, feature selection, PCA, and undersampling were all fit only on the training folds to avoid data leakage.
