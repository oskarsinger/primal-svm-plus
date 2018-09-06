# Primal-Only, Unconstrained SVM+ for Fast Estimation with Privileged Information

## What are Privileged Information and SVM+?
In the typical supervised learning scenario, all features that are present at training must also be available at test time. But what if we have some subset of features that is 'supervised', i.e. associated with a label, but is only available at training? We term this subset "privileged information". We term the machine learning scenario that takes advantage of privileged information "learning using privileged information" or LUPI.

Currently, the only way to involve privileged information into a machine learning setting is through SVM+, an extension of SVM that replaces the conventional SVM slack variables with a parameterized function of the privileged information. 

## Why Primal-Only, Unconstrained?
The original SVM+ optimization problem must be solved in the dual, for which the quadratic form is necessarily low-rank. Additionally, the SVM+ model most commonly used basically makes the equivalent assumption to linear separability in a classical SVM. Additional slacks must be introduced to ensure existence of feasible solution to any realistic data set. A version with additional slacks does exist, but the chosen slacks still require solving the poorly conditioned dual. 

We introduce a choice of additional slack that allows us to reformulate the optimization problem using explicit hinge losses and no constraints. Not only is this a more intuitive and simpler formulation of the problem, it also allows us to use (stochastic) subgradient descent with no necessary projections, circumventing the difficulties of the dual problem and giving lightnight fast convergence to a meaningful solution to the parameter estimation problem.

## Code

Dependencies:
 * click
 * numpy
 * scipy
 * pathos

### Loaders
These classes are intended to load data from files or generate synthetic data, usually randomized. They do light pre-processing to the extent that the output can be treated somewhat agnostically across loaders. The more standard loaders (those not used for interactive scenarios like bandit problems and reinforcement learning) provide the following functions:

* `get_data`: no arguments, returns a collection of data matrices
* `rows`: number of unique rows (i.e. number of samples) that the loader has served up to now
* `cols`: number of columns (i.e. number of features) of the data served by the loader
* `name`: unique string identifier for the loader

#### Synthetic
These classes are for implementations of human-specified data-generating processes, usually random but sometimes adversarial.

##### Gaussian
This generates a Gaussian matrix. Arguments:
 * `n`: number of rows
 * `p`: number of cols
 * `lazy`: (default=`True`) if `True`, generates and stores matrix only at first call of `get_data`, else generates and stores matrix upon creation of instance
 * `k`: rank of data matrix
 * `mean`: mean of the matrix rows

##### Bernoulli
This generates a matrix of i.i.d. Bernoulli samples. Arguments:
 * `n`: number of rows
 * `m`: number of cols
 * `p`: (default=`0.5`) success probability of the Bernoulli distribution used to sample entries
 * `lazy`: (default=`True`) if `True`, generates and stores matrix only at first call of `get_data`, else generates and stores matrix upon creation of instance

##### Linear Dynamics Sequence
This generates a matrix in which row _i_ is a rotation of row _i-1_. Arguments:
 * `A`: rotation matrix
 * `num_data`: number of data points to produce
 * `seed`: (default=a column vector of all 1s) row 0 of the data matrix is set to `A` times the seed

#### Real
These classes are intended to pull in actual ARDS data sets that were collected and curated and stored in a CSV. I no longer have access to the data, so they are just here for illustration purposes.

##### LUPI

##### No LUPI



### Servers

#### Batch

#### Minibatch

### Models
These classes are used to specify the details (and limited state to what extent is necessary) of a machine learning parameter estimation problem necessary for an abstracted first-order optimizer. Each class offers the following functions:

 * `get_gradient`: takes some data and parameter state, then returns the gradient of objective function as defined in terms of the input data, evaluated at the input parameter state
 * `get_objective`: takes some data and parameter state, then returns the objective function as defined in terms of the input data, evaluated at the input parameter state
 * `get_residuals`: takes some data and parameter state, then returns a vector of the evaluation of the objective sans regularization on each data point 
 * `get_datum`: takes some data and a row index and returns the data point at that row index
 * `get_projected`: takes some data and parameter state and returns the parameters projected onto whatever feasible region is specified by the parameter estimation problem

#### Pegasos SVM

#### Li et al. 2016 SVM+

#### Pegasos SVM+

### Optimizers
These classes are used to solve optimization problems. They are typically given blackbox objective, gradient, and projection functions that take optimization variable states as input. Each class offers the following functions:

 * `run`: runs the optimization algorithm and sets the internal parameter state
 * `get_parameters`: returns the current value of the parameters, throws exception if hasn't been run yet

#### Full Matrix AdaGrad

#### Adam

#### Stochastic Coordinate Adam

#### BFGS

#### HyperBand

### Testers
