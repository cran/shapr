% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/approach_gaussian.R
\name{create_marginal_data_gaussian}
\alias{create_marginal_data_gaussian}
\title{Generate marginal Gaussian data using Cholesky decomposition}
\usage{
create_marginal_data_gaussian(n_MC_samples, Sbar_features, mu, cov_mat)
}
\arguments{
\item{n_MC_samples}{Integer. The number of samples to generate.}

\item{Sbar_features}{Vector of integers indicating which marginals to sample from.}

\item{mu}{Numeric vector containing the expected values for all features in the multivariate Gaussian distribution.}

\item{cov_mat}{Numeric matrix containing the covariance between all features
in the multivariate Gaussian distribution.}
}
\description{
Given a multivariate Gaussian distribution, this function creates data from specified marginals of said distribution.
}
\author{
Lars Henry Berge Olsen
}
\keyword{internal}
