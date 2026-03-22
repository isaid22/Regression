## Lasso Regression

This technique is used in case of variable selection for a regression problem. The dataset used in this example is the diabetes dataset. This dataset is available in `sklearn` library. In this version of the dataset, the feature columns are:

```bash
['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
```

and the target variable and its type is 

```bash
Name: target, dtype: float64
```

The objective is to find which features are important and should be included for a regression model to predict the target variable. The method of feature selection is Lasso regression. In such method, each feature is tested for its importance to the target regresion value. This requires the evaulation of a cost function. 

### Lasso Regression Cost Function

The objective function for Lasso (Least Absolute Shrinkage and Selection Operator) Regression combines the Mean Squared Error (MSE) loss with an $L_1$ regularization penalty:

$$
J(w) = \frac{1}{2n} \sum_{i=1}^{n} (y_i - w^T x_i)^2 + \lambda \sum_{j=1}^{p} |w_j|
$$

**Where:**
* $n$ is the number of samples (observations).
* $p$ is the number of features.
* $y_i$ is the actual target value for the $i$-th sample.
* $x_i$ is the feature vector for the $i$-th sample.
* $w$ represents the model weights (coefficients), and $w_j$ is the weight of the $j$-th feature.
* $\lambda$ (or $\alpha$) is the regularization parameter that controls the penalty's strength.

For each variable, the coefficient shrinks to zero to test whether or not the cost function has been impacted.

This example uses `LassoLarsCV` method.  This method compute the entire sloution path as $\lambda$ ( sometimes knowns as $\alpha$) decreases. The process is as follow:

1. Start with all coefficients = 0
2. Find the feathre most correlated with target
3. Increase its coefficient
4. When another feature becomes equally correlated, include it
5. Continue in a piecewise linear path

Training Data is split into five partitions. This is indicated by `cv=5` in the model definition as shown in [lsso-fit.py](./lasso-fit.py). There is a partition `X_test` that is hold-out data not used for training.


# Model definition
The model is defined as a pipeline:

```python
model = make_pipeline(
    StandardScaler(),
    LassoLarsCV(cv=5)
)
```

This pipeline includes a standardization step `StandardScaler`. This obkect standardizes each feature (zero mean, unit varianve), and a Lasso regression model model that stores best $\alpha$ and the final learned coefficients and intercept.

## Save the trained model

The model trained can be saved with `joblib` library. This is done in [lsso-fit.py](./lasso-fit.py) through 

```python
joblib.dump(model, "lasso_diabetes_pipeline.joblib")
```

## Training

The training script, when executed, will produce the following output:

```bash
eatures:
['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
Target
target
count    442.000000
mean     152.133484
std       77.093005
min       25.000000
25%       87.000000
50%      140.500000
75%      211.500000
max      346.000000
Name: target, dtype: float64
Chosen alpha: 0.11232208318658762
Test RMSE: 53.27219745603079
Test R^2: 0.4867836848878935

Coefficients:
s1    -31.435809
s5     28.660745
bmi    25.312431
bp     18.059671
s2     14.631624
sex   -11.267241
s4     11.033715
age     2.170760
s6      1.283755
s3      0.000000
dtype: float64

Selected features (non-zero coefficients):
s1    -31.435809
s5     28.660745
bmi    25.312431
bp     18.059671
s2     14.631624
sex   -11.267241
s4     11.033715
age     2.170760
s6      1.283755
dtype: float64
Predictions for new data:
[138.26084876 182.83459068 132.54624545 292.69766472 124.12390198]
```
This shows that feature `s3` is not selected.



This save the entire pipeline in the current directory.

## Inference

Use `joblib` library to reload the saved pipeline back into runtime for inference. This is done through:

```python
model = joblib.load("lasso_diabetes_pipeline.joblib")
```

and the example is shown in [reload-inference.py](./reload-inference.py.)

