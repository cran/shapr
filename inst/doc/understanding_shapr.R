## ---- include = FALSE----------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  fig.width = 7,
  fig.height = 3
)

## ----setup, include=FALSE, warning=FALSE---------------------------------
library(shapr)

## ---- warning=FALSE------------------------------------------------------
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

## ------------------------------------------------------------------------
# Use the Gaussian approach
explanation_gaussian <- explain(
  x_test,
  approach = "gaussian",
  explainer = explainer,
  prediction_zero = p
)

# Plot the resulting explanations for observations 1 and 6
plot(explanation_gaussian, plot_phi0 = FALSE, index_x_test = c(1, 6))

## ------------------------------------------------------------------------
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

## ------------------------------------------------------------------------
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

## ------------------------------------------------------------------------
library(gbm)

form <- as.formula(paste0(y_var, "~", paste0(x_var, collapse = "+")))
xy_train <- data.frame(x_train, medv = y_train)

# Fitting a gbm model
set.seed(825)
model <- gbm::gbm(
  form,
  data = xy_train,
  distribution = "gaussian"
  )

# Create custom function of model_type for gbm
model_type.gbm <- function(x) {
  ifelse(
    x$distribution$name %in% c("bernoulli", "adaboost"),
    "classification",
    "regression"
  )
}

# Create custom function of predict_model for gbm
predict_model.gbm <- function(x, newdata) {

  if (!requireNamespace('gbm', quietly = TRUE)) {
    stop('The gbm package is required for predicting train models')
  }
  model_type <- model_type(x)

  if (model_type == "classification") {

    predict(x, as.data.frame(newdata), type = "response", n.trees = x$n.trees)
  } else {

    predict(x, as.data.frame(newdata), n.trees = x$n.trees)
  }
}

# Prepare the data for explanation
set.seed(123)
explainer <- shapr(xy_train, model, feature_labels = x_var)

# Spedifying the phi_0, i.e. the expected prediction without any features
p0 <- mean(xy_train[, y_var])

# Computing the actual Shapley values with kernelSHAP accounting for feature dependence using
# the empirical (conditional) distribution approach with
# bandwidth parameter sigma = 0.1 (default)
explanation <- explain(
  x_test,
  explainer,
  approach = "empirical",
  prediction_zero = p0
)

# Plot the resulting explanations for observations 1 and 6, excluding
# the no-covariate effect.
plot(explanation_combined, plot_phi0 = FALSE, index_x_test = c(1, 6))

