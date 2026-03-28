# Derivation of Beta Coefficient for Ordinary Least Squares (OLS) Linear Regression

Here is the step-by-step derivation of the $\beta$ coefficient vector for Ordinary Least Squares (OLS) linear regression using linear algebra.

## 1. Define the Model and Dimensions

The generic multiple linear regression model in vectorized format is:

$$y = X\beta + \varepsilon$$

Given your notation:
*   $n$: Number of observations.
*   $p$: Number of variables (dimensions/features in $X$).
*   $y$: An $n \times 1$ column vector of the observed target values.
*   $X$: An $n \times p$ design matrix of our variables $[x_1, x_2, \dots, x_p]$.
*   $\beta$: A $p \times 1$ column vector of the unknown coefficients we want to estimate.
*   $\varepsilon$: An $n \times 1$ column vector of the error terms (residuals).

## 2. Define the Objective Function

In OLS regression, the goal is to find the $\beta$ that minimizes the Sum of Squared Residuals (SSR). The residual vector is:

$$\varepsilon = y - X\beta$$

The Sum of Squared Residuals is the dot product of the residual vector with itself, which can be written using the transpose notation ($\varepsilon^T \varepsilon$):

$$SSR = \varepsilon^T \varepsilon = (y - X\beta)^T (y - X\beta)$$

## 3. Expand the Objective Function

Using the distributive property of matrix transposition — noting that $(AB)^T = B^T A^T$ — we expand the equation:

$$SSR = (y^T - (X\beta)^T)(y - X\beta)$$
$$SSR = (y^T - \beta^T X^T)(y - X\beta)$$
$$SSR = y^T y - y^T X \beta - \beta^T X^T y + \beta^T X^T X \beta$$

Notice the two middle terms: $y^T X \beta$ and $\beta^T X^T y$. 
Let's look at their dimensions: $(1 \times n) \times (n \times p) \times (p \times 1) = (1 \times 1)$. 
Because they result in a $1 \times 1$ scalar, a scalar is equal to its own transpose. Thus:

$$(y^T X \beta)^T = \beta^T X^T y$$

This allows us to combine the two middle terms:

$$SSR = y^T y - 2 \beta^T X^T y + \beta^T X^T X \beta$$

## 4. Differentiate with Respect to $\beta$

To find the minimum SSR, we take the partial derivative (gradient) of the SSR function with respect to the vector $\beta$ and set it to a zero vector ($0$).

Using standard matrix calculus rules:
*   The derivative of a constant $y^T y$ with respect to $\beta$ is $0$.
*   The derivative of $-2 \beta^T X^T y$ with respect to $\beta$ is $-2 X^T y$.
*   The derivative of the quadratic form $\beta^T X^T X \beta$ with respect to $\beta$ is $2 X^T X \beta$.

Applying these rules gives us the gradient:

$$\frac{\partial (SSR)}{\partial \beta} = 0 - 2 X^T y + 2 X^T X \beta$$

## 5. Set to Zero and Solve for $\beta$

Set the derivative equal to $0$ to find the optimal $\beta$ (which we denote as $\hat{\beta}$ to signify it's an estimate):

$$-2 X^T y + 2 X^T X \hat{\beta} = 0$$

Divide by 2 and move the term without $\hat{\beta}$ to the right side. This gives us the **Normal Equations**:

$$X^T X \hat{\beta} = X^T y$$

## 6. Isolate $\hat{\beta}$ using the Matrix Inverse

To solve for $\hat{\beta}$, we need to multiply both sides of the equation by the inverse of $(X^T X)$. 

*(Note: $(X^T X)$ is a $p \times p$ square matrix. For the inverse $(X^T X)^{-1}$ to exist, $X$ must have full column rank, meaning no variables in $X$ are perfectly collinear, and we must have $n \ge p$.)*

Multiply both sides by $(X^T X)^{-1}$ on the left:

$$(X^T X)^{-1} (X^T X) \hat{\beta} = (X^T X)^{-1} X^T y$$

Since a matrix multiplied by its inverse results in the Identity matrix ($I$), and $I\hat{\beta} = \hat{\beta}$, we arrive at the final solution for the beta coefficient vector:

$$\hat{\beta} = (X^T X)^{-1} X^T y$$
