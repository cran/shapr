% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/sampling.R
\name{sample_gaussian}
\alias{sample_gaussian}
\title{Sample conditional Gaussian variables}
\usage{
sample_gaussian(index_given, n_samples, mu, cov_mat, m, x_test)
}
\arguments{
\item{index_given}{Integer vector. The indices of the features to condition upon. Note that
\code{min(index_given) >= 1} and \code{max(index_given) <= m}.}

\item{m}{Positive integer. The total number of features.}

\item{x_test}{Numeric matrix. Contains the features of the observation whose
predictions ought to be explained (test data).}
}
\value{
data.table
}
\description{
Sample conditional Gaussian variables
}
\author{
Martin Jullum
}
\keyword{internal}
