% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{prepare_data_copula_cpp_caus}
\alias{prepare_data_copula_cpp_caus}
\title{Generate (Gaussian) Copula MC samples for the causal setup with a single MC sample for each explicand}
\usage{
prepare_data_copula_cpp_caus(
  MC_samples_mat,
  x_explain_mat,
  x_explain_gaussian_mat,
  x_train_mat,
  S,
  mu,
  cov_mat
)
}
\arguments{
\item{MC_samples_mat}{arma::mat.
Matrix of dimension (\code{n_MC_samples}, \code{n_features}) containing samples from the univariate standard normal.}

\item{x_explain_mat}{arma::mat.
Matrix of dimension (\code{n_explain}, \code{n_features}) containing the observations to explain.}

\item{x_explain_gaussian_mat}{arma::mat.
Matrix of dimension (\code{n_explain}, \code{n_features}) containing the observations to explain after being transformed
using the Gaussian transform, i.e., the samples have been transformed to a standardized normal distribution.}

\item{x_train_mat}{arma::mat.
Matrix of dimension (\code{n_train}, \code{n_features}) containing the training observations.}

\item{S}{arma::mat.
Matrix of dimension (\code{n_coalitions}, \code{n_features}) containing binary representations of the used coalitions.
S cannot contain the empty or grand coalition, i.e., a row containing only zeros or ones.
This is not a problem internally in shapr as the empty and grand coalitions are treated differently.}

\item{mu}{arma::vec.
Vector of length \code{n_features} containing the mean of each feature after being transformed using the Gaussian
transform, i.e., the samples have been transformed to a standardized normal distribution.}

\item{cov_mat}{arma::mat.
Matrix of dimension (\code{n_features}, \code{n_features}) containing the pairwise covariance between all pairs of features
after being transformed using the Gaussian transform, i.e., the samples have been transformed to a standardized
normal distribution.}
}
\value{
An arma::cube/3D array of dimension (\code{n_MC_samples}, \code{n_explain} * \code{n_coalitions}, \code{n_features}), where
the columns (\emph{,j,}) are matrices of dimension (\code{n_MC_samples}, \code{n_features}) containing the conditional Gaussian
copula MC samples for each explicand and coalition on the original scale.
}
\description{
Generate (Gaussian) Copula MC samples for the causal setup with a single MC sample for each explicand
}
\author{
Lars Henry Berge Olsen
}
\keyword{internal}
