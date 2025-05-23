% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{hat_matrix_cpp}
\alias{hat_matrix_cpp}
\title{Computing single H matrix in AICc-function using the Mahalanobis distance}
\usage{
hat_matrix_cpp(X, mcov, S_scale_dist, h)
}
\arguments{
\item{X}{matrix.}

\item{mcov}{matrix
The covariance matrix of X.}

\item{S_scale_dist}{logical.
Indicating whether the Mahalanobis distance should be scaled with the number of variables}

\item{h}{numeric specifying the scaling (sigma)}
}
\value{
Matrix of dimension \code{ncol(X)*ncol(X)}
}
\description{
Computing single H matrix in AICc-function using the Mahalanobis distance
}
\author{
Martin Jullum
}
\keyword{internal}
