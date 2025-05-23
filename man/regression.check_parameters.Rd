% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/approach_regression_separate.R
\name{regression.check_parameters}
\alias{regression.check_parameters}
\title{Check regression parameters}
\usage{
regression.check_parameters(internal)
}
\arguments{
\item{internal}{List.
Holds all parameters, data, functions and computed objects used within \code{\link[=explain]{explain()}}
The list contains one or more of the elements \code{parameters}, \code{data}, \code{objects}, \code{iter_list}, \code{timing_list},
\code{main_timing_list}, \code{output}, and \code{iter_timing_list}.}
}
\value{
The same \code{internal} list, but added logical indicator \code{internal$parameters$regression.tune}
if we are to tune the regression model/models.
}
\description{
Check regression parameters
}
\author{
Lars Henry Berge Olsen
}
\keyword{internal}
