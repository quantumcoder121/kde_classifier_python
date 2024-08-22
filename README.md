# Classification using Kernel Density Estimation

## Theory behind the Classifier

Consider we have $n$ classes in our classification problem. For each class $0 \leq i < n$, let the set of features for the training set be $X_i$. We first fit $n$ Kernel Density Estimators (KDEs) on these $n$ sets. Let the fitted KDE for the $n^{th}$ class be $KDE_n$.

Now, while finding the class $y$ of a test sample $x$, we consider the following.

$$
P[y = i | x] = \frac{P[x | y = i] \cdot P[y = i]}{\Sigma_{i = 0}^{n - 1} P[x | y = j] \cdot P[y = j]}
$$

Here, we are approximating $P[x | y = i]$ by $KDE_i(x)$ and $P[y = i]$ is calculated empirically from the training dataset. Thus,

$$
P[y = i | x] = \frac{KDE_i(x) \cdot P[y = i]}{\Sigma_{i = 0}^{n - 1} KDE_j(x) \cdot P[y = j]}
$$
