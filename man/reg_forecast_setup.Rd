% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/explain_forecast.R
\name{reg_forecast_setup}
\alias{reg_forecast_setup}
\title{Set up exogenous regressors for explanation in a forecast model.}
\usage{
reg_forecast_setup(x, horizon, group)
}
\arguments{
\item{x}{A matrix with the exogenous variables.}

\item{horizon}{Numeric.
The forecast horizon to explain. Passed to the \code{predict_model} function.}

\item{group}{The list of endogenous groups, to append exogenous groups to.}
}
\value{
A list containing
\itemize{
\item fcast A matrix containing the exogenous observations needed for each observation.
\item group The list group with the exogenous groups appended.
}
}
\description{
Set up exogenous regressors for explanation in a forecast model.
}
\keyword{internal}
