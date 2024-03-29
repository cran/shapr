% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/observations.R
\name{observation_impute}
\alias{observation_impute}
\title{Generate permutations of training data using test observations}
\usage{
observation_impute(
  W_kernel,
  S,
  x_train,
  x_test,
  w_threshold = 0.7,
  n_samples = 1000
)
}
\arguments{
\item{W_kernel}{Numeric matrix. Contains all nonscaled weights between training and test
observations for all feature combinations. The dimension equals \code{n_train x m}.}

\item{S}{Integer matrix of dimension \code{n_combinations x m}, where \code{n_combinations}
and \code{m} equals the total number of sampled/non-sampled feature combinations and
the total number of unique features, respectively. Note that \code{m = ncol(x_train)}.}

\item{x_train}{Numeric matrix}

\item{x_test}{Numeric matrix}

\item{w_threshold}{Numeric vector of length 1, where \code{w_threshold > 0} and
\code{w_threshold <= 1}. If \code{w_threshold = .8} we will choose the \code{K} samples with
the largest weight so that the sum of the weights accounts for 80\% of the total weight.}
}
\value{
data.table
}
\description{
Generate permutations of training data using test observations
}
\author{
Nikolai Sellereite
}
\keyword{internal}
