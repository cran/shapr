% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/setup.R
\name{get_output_args_default}
\alias{get_output_args_default}
\title{Gets the default values for the output arguments}
\usage{
get_output_args_default(
  keep_samp_for_vS = FALSE,
  MSEv_uniform_comb_weights = TRUE,
  saving_path = tempfile("shapr_obj_", fileext = ".rds")
)
}
\arguments{
\item{keep_samp_for_vS}{Logical.
Indicates whether the samples used in the Monte Carlo estimation of v_S should be returned (in \code{internal$output}).
Not used for \code{approach="regression_separate"} or \code{approach="regression_surrogate"}.}

\item{MSEv_uniform_comb_weights}{Logical.
If \code{TRUE} (default), then the function weights the coalitions uniformly when computing the MSEv criterion.
If \code{FALSE}, then the function use the Shapley kernel weights to weight the coalitions when computing the MSEv
criterion.
Note that the Shapley kernel weights are replaced by the sampling frequency when not all coalitions are considered.}

\item{saving_path}{String.
The path to the directory where the results of the iterative estimation procedure should be saved.
Defaults to a temporary directory.}
}
\value{
A list of default output arguments.
}
\description{
Gets the default values for the output arguments
}
\author{
Martin Jullum
}
