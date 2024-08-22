# Classification using Kernel Density Estimation

## Theory behind the Classifier

Consider we have $n$ classes in our classification problem. For each class $0 \leq i < n$, let the set of features for the training set be $X_i$. We first fit $n$ Kernel Density Estimators (KDEs) on these $n$ sets. Let the fitted KDE for the $n^{th}$ class be $KDE_n$.

Now, while finding the class $y$ of a test sample $x$, we consider the following.

$
P[y = i | x] = 
$
