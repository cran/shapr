% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/model.R, R/model_ar.R, R/model_arima.R,
%   R/model_glm.R, R/model_lm.R, R/model_mgcv_gam.R, R/model_ranger.R,
%   R/model_workflow.R, R/model_xgboost.R
\name{predict_model}
\alias{predict_model}
\alias{predict_model.default}
\alias{predict_model.ar}
\alias{predict_model.Arima}
\alias{predict_model.forecast_ARIMA}
\alias{predict_model.glm}
\alias{predict_model.lm}
\alias{predict_model.gam}
\alias{predict_model.ranger}
\alias{predict_model.workflow}
\alias{predict_model.xgb.Booster}
\title{Generate predictions for input data with specified model}
\usage{
predict_model(x, newdata, ...)

\method{predict_model}{default}(x, newdata, ...)

\method{predict_model}{ar}(x, newdata, newreg, horizon, ...)

\method{predict_model}{Arima}(
  x,
  newdata,
  newreg,
  horizon,
  explain_idx,
  explain_lags,
  y,
  xreg,
  ...
)

\method{predict_model}{forecast_ARIMA}(x, newdata, newreg, horizon, ...)

\method{predict_model}{glm}(x, newdata, ...)

\method{predict_model}{lm}(x, newdata, ...)

\method{predict_model}{gam}(x, newdata, ...)

\method{predict_model}{ranger}(x, newdata, ...)

\method{predict_model}{workflow}(x, newdata, ...)

\method{predict_model}{xgb.Booster}(x, newdata, ...)
}
\arguments{
\item{x}{Model object for the model to be explained.}

\item{newdata}{A data.frame/data.table with the features to predict from.}

\item{...}{\code{newreg} and \code{horizon} parameters used in models passed to \verb{[explain_forecast()]}}

\item{horizon}{Numeric.
The forecast horizon to explain. Passed to the \code{predict_model} function.}

\item{explain_idx}{Numeric vector.
The row indices in data and reg denoting points in time to explain.}

\item{y}{Matrix, data.frame/data.table or a numeric vector.
Contains the endogenous variables used to estimate the (conditional) distributions
needed to properly estimate the conditional expectations in the Shapley formula
including the observations to be explained.}

\item{xreg}{Matrix, data.frame/data.table or a numeric vector.
Contains the exogenous variables used to estimate the (conditional) distributions
needed to properly estimate the conditional expectations in the Shapley formula
including the observations to be explained.
As exogenous variables are used contemporaneously when producing a forecast,
this item should contain nrow(y) + horizon rows.}
}
\value{
Numeric. Vector of size equal to the number of rows in \code{newdata}.
}
\description{
Performs prediction of response
\code{\link[stats:lm]{stats::lm()}},
\code{\link[stats:glm]{stats::glm()}},
\code{\link[ranger:ranger]{ranger::ranger()}},
\code{\link[mgcv:gam]{mgcv::gam()}},
\code{\link[workflows:workflow]{workflows::workflow()}} (i.e., \code{tidymodels} models), and
\code{\link[xgboost:xgb.train]{xgboost::xgb.train()}} with binary or continuous
response. See details for more information.
}
\details{
The following models are currently supported:
\itemize{
\item \code{\link[stats:lm]{stats::lm()}}
\item \code{\link[stats:glm]{stats::glm()}}
\item \code{\link[ranger:ranger]{ranger::ranger()}}
\item \code{\link[mgcv:gam]{mgcv::gam()}}
\item \code{\link[workflows:workflow]{workflows::workflow()}}
\item \code{\link[xgboost:xgb.train]{xgboost::xgb.train()}}
}

If you have a binary classification model we'll always return the probability prediction
for a single class.

If you are explaining a model not supported natively, you need to create the \verb{[predict_model()]} function yourself,
and pass it on to as an argument to \verb{[explain()]}.

For more details on how to explain such non-supported models (i.e. custom models), see the Advanced usage section
of the general usage: \cr
From R: \code{vignette("general_usage", package = "shapr")}  \cr
Web: \url{https://norskregnesentral.github.io/shapr/articles/general_usage.html#explain-custom-models}
}
\examples{
# Load example data
data("airquality")
airquality <- airquality[complete.cases(airquality), ]
# Split data into test- and training data
x_train <- head(airquality, -3)
x_explain <- tail(airquality, 3)
# Fit a linear model
model <- lm(Ozone ~ Solar.R + Wind + Temp + Month, data = x_train)

# Predicting for a model with a standardized format
predict_model(x = model, newdata = x_explain)
}
\author{
Martin Jullum
}
\keyword{internal}
