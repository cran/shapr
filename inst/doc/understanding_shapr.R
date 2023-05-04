## ---- include = FALSE---------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  fig.width = 7,
  fig.height = 3
)

## ----setup, include=FALSE, warning=FALSE--------------------------------------
library(shapr)

## ---- warning=FALSE-----------------------------------------------------------
library(xgboost)
library(shapr)

data("Boston", package = "MASS")

x_var <- c("lstat", "rm", "dis", "indus")
y_var <- "medv"

x_train <- as.matrix(Boston[-1:-6, x_var])
y_train <- Boston[-1:-6, y_var]
x_test <- as.matrix(Boston[1:6, x_var])

# Fitting a basic xgboost model to the training data
model <- xgboost(
  data = x_train,
  label = y_train,
  nround = 20,
  verbose = FALSE
)

# Prepare the data for explanation
explainer <- shapr(x_train, model)

# Specifying the phi_0, i.e. the expected prediction without any features
p <- mean(y_train)

# Computing the actual Shapley values with kernelSHAP accounting for feature dependence using
# the empirical (conditional) distribution approach with bandwidth parameter sigma = 0.1 (default)
explanation <- explain(
  x_test,
  approach = "empirical",
  explainer = explainer,
  prediction_zero = p
)

# Printing the Shapley values for the test data.
# For more information about the interpretation of the values in the table, see ?shapr::explain.
print(explanation$dt)

# Plot the resulting explanations for observations 1 and 6
plot(explanation, plot_phi0 = FALSE, index_x_test = c(1, 6))

## -----------------------------------------------------------------------------
# Use the Gaussian approach
explanation_gaussian <- explain(
  x_test,
  approach = "gaussian",
  explainer = explainer,
  prediction_zero = p
)

# Plot the resulting explanations for observations 1 and 6
plot(explanation_gaussian, plot_phi0 = FALSE, index_x_test = c(1, 6))

## -----------------------------------------------------------------------------
# Use the Gaussian copula approach
explanation_copula <- explain(
  x_test,
  approach = "copula",
  explainer = explainer,
  prediction_zero = p
)

# Plot the resulting explanations for observations 1 and 6, excluding
# the no-covariate effect
plot(explanation_copula, plot_phi0 = FALSE, index_x_test = c(1, 6))

## -----------------------------------------------------------------------------
# Use the conditional inference tree approach
explanation_ctree <- explain(
  x_test,
  approach = "ctree",
  explainer = explainer,
  prediction_zero = p
)

# Plot the resulting explanations for observations 1 and 6, excluding 
# the no-covariate effect
plot(explanation_ctree, plot_phi0 = FALSE, index_x_test = c(1, 6))

## -----------------------------------------------------------------------------
x_var_cat <- c("lstat", "chas", "rad", "indus")
y_var <- "medv"

# convert to factors
Boston$rad = as.factor(Boston$rad)
Boston$chas = as.factor(Boston$chas)

x_train_cat <- Boston[-1:-6, x_var_cat]
y_train <- Boston[-1:-6, y_var]
x_test_cat <- Boston[1:6, x_var_cat]

# -- special function when using categorical data + xgboost
dummylist <- make_dummies(traindata = x_train_cat, testdata = x_test_cat)

x_train_dummy <- dummylist$train_dummies
x_test_dummy <- dummylist$test_dummies

# Fitting a basic xgboost model to the training data
model_cat <- xgboost::xgboost(
  data = x_train_dummy,
  label = y_train,
  nround = 20,
  verbose = FALSE
)
model_cat$feature_list <- dummylist$feature_list

explainer_cat <- shapr(dummylist$traindata_new, model_cat)

p <- mean(y_train)

explanation_cat <- explain(
  dummylist$testdata_new,
  approach = "ctree",
  explainer = explainer_cat,
  prediction_zero = p
)

# Plot the resulting explanations for observations 1 and 6, excluding
# the no-covariate effect
plot(explanation_cat, plot_phi0 = FALSE, index_x_test = c(1, 6))


## -----------------------------------------------------------------------------
# Use the conditional inference tree approach
# We can specify parameters used to building trees by specifying mincriterion, 
# minsplit, minbucket

explanation_ctree <- explain(
  x_test,
  approach = "ctree",
  explainer = explainer,
  prediction_zero = p,
  mincriterion = 0.80, 
  minsplit = 20,
  minbucket = 20
)

# Default parameters (based on (Hothorn, 2006)) are:
# mincriterion = 0.95
# minsplit = 20
# minbucket = 7

## -----------------------------------------------------------------------------
# Use the conditional inference tree approach
# Specify a vector of mincriterions instead of just one
# In this case, when conditioning on 1 or 2 features, use mincriterion = 0.25
# When conditioning on 3 or 4 features, use mincriterion = 0.95

explanation_ctree <- explain(
  x_test,
  approach = "ctree",
  explainer = explainer,
  prediction_zero = p,
  mincriterion = c(0.25, 0.25, 0.95, 0.95)
)

## -----------------------------------------------------------------------------
# Use the combined approach
explanation_combined <- explain(
  x_test,
  approach = c("empirical", "copula", "gaussian", "gaussian"),
  explainer = explainer,
  prediction_zero = p
)

# Plot the resulting explanations for observations 1 and 6, excluding
# the no-covariate effect
plot(explanation_combined, plot_phi0 = FALSE, index_x_test = c(1, 6))

## -----------------------------------------------------------------------------
# Use the combined approach
explanation_combined <- explain(
  x_test,
  approach = c("ctree", "ctree", "ctree", "empirical"),
  explainer = explainer,
  prediction_zero = p
)

## -----------------------------------------------------------------------------
x_var <- c("lstat", "rm", "dis", "indus")
y_var <- "medv"

# Convert two features as factors
dt <- Boston[, c(x_var, y_var)]
dt$rm <- as.factor(round(dt$rm/3))
dt$dis <- as.factor(round(dt$dis/4))

xy_train_cat <- dt[-1:-6, ]
y_train_cat <- dt[-1:-6, y_var]
x_train_cat <- dt[-1:-6, x_var]
x_test_cat <- dt[1:6, x_var]


# Fit a basic linear regression model to the training data
model <- lm(medv ~ lstat + rm + dis + indus, data = xy_train_cat)

# Prepare the data for explanation
explainer <- shapr(x_train_cat, model)

# Specifying the phi_0, i.e. the expected prediction without any features
p <- mean(y_train_cat)

# Computing the actual Shapley values with kernelSHAP accounting for feature dependence using
explanation_categorical <- explain(
  x_test_cat,
  approach = "ctree",
  explainer = explainer,
  prediction_zero = p
)
# Note that nothing has to be specified to tell "ctree" that two of the features are 
# cateogrical and two are numerical

# Plot the resulting explanations for observations 1 and 6, excluding 
# the no-covariate effect
plot(explanation_categorical, plot_phi0 = FALSE, index_x_test = c(1, 6))


## -----------------------------------------------------------------------------
library(gbm)

xy_train <- data.frame(x_train,medv = y_train)

form <- as.formula(paste0(y_var,"~",paste0(x_var,collapse="+")))

# Fitting a gbm model
set.seed(825)
model <- gbm::gbm(
  form,
  data = xy_train,
  distribution = "gaussian"
)

#### Full feature versions of the three required model functions ####

predict_model.gbm <- function(x, newdata) {
  
  if (!requireNamespace('gbm', quietly = TRUE)) {
    stop('The gbm package is required for predicting train models')
  }

  model_type <- ifelse(
    x$distribution$name %in% c("bernoulli","adaboost"),
    "classification",
    "regression"
  )
  if (model_type == "classification") {

    predict(x, as.data.frame(newdata), type = "response",n.trees = x$n.trees)
  } else {

    predict(x, as.data.frame(newdata),n.trees = x$n.trees)
  }
}

get_model_specs.gbm <- function(x){
  feature_list = list()
  feature_list$labels <- labels(x$Terms)
  m <- length(feature_list$labels)

  feature_list$classes <- attr(x$Terms,"dataClasses")[-1]
  feature_list$factor_levels <- setNames(vector("list", m), feature_list$labels)
  feature_list$factor_levels[feature_list$classes=="factor"] <- NA # the model object doesn't contain factor levels info

  return(feature_list)
}

# Prepare the data for explanation
set.seed(123)
explainer <- shapr(xy_train, model)
p0 <- mean(xy_train[,y_var])
explanation <- explain(x_test, explainer, approach = "empirical", prediction_zero = p0)
# Plot results
plot(explanation)

#### Minimal version of the three required model functions ####      
# Note: Working only for this exact version of the model class 
# Avoiding to define get_model_specs skips all feature         
# consistency checking between your data and model             

# Removing the previously defined functions to simulate a fresh start
rm(predict_model.gbm)
rm(get_model_specs.gbm)

predict_model.gbm <- function(x, newdata) {
    predict(x, as.data.frame(newdata),n.trees = x$n.trees)
}

# Prepare the data for explanation
set.seed(123)
explainer <- shapr(x_train, model)
p0 <- mean(xy_train[,y_var])
explanation <- explain(x_test, explainer, approach = "empirical", prediction_zero = p0)
# Plot results
plot(explanation)



