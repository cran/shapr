% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/predictions.R
\name{prediction}
\alias{prediction}
\title{Calculate Shapley weights for test data}
\usage{
prediction(dt, prediction_zero, explainer)
}
\arguments{
\item{dt}{data.table}

\item{prediction_zero}{Numeric. The value to use for \code{phi_0}.}

\item{explainer}{An object of class \code{explainer}. See \code{\link{shapr}}.}
}
\value{
An object of class \code{c("shapr", "list")}. For more details see \code{\link{explain}}.
}
\description{
This function should only be called internally, and not be used as
a stand-alone function.
}
\details{
If \code{dt} does not contain three columns called \code{id}, \code{id_combination} and \code{w}
the function will fail. \code{id} represents a unique key for a given test observation,
and \code{id_combination} is a unique key for which feature combination the row represents. \code{w}
represents the Shapley value of feature combination given by \code{id_combination}. In addition
to these three columns, \code{dt} should also have columns which matches the variables used
when training the model.

I.e. you have fitted a linear model using the features \code{x1},
\code{x2} and \code{x3}, and you want to explain 5 test observations using the exact method, i.e.
setting \code{exact = TRUE} in \code{\link{shapr}}, the following properties should be satisfied
\enumerate{
\item \code{colnames(dt)} equals \code{c("x1", "x2", "x3", "id", "id_combination", ""w)}
\item \code{dt[, max(id)]} equals the number of test observations
\item \code{dt[, min(id)]} equals 1L.
\item \code{dt[, max(id_combination)]} equals \code{2^m} where m equals the number of features.
\item \code{dt[, min(id_combination)]} equals 1L.
\item \code{dt[, type(w)]} equals \code{double}.
}
}
\author{
Nikolai Sellereite
}
\keyword{internal}
