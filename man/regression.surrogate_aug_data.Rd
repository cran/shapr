% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/approach_regression_surrogate.R
\name{regression.surrogate_aug_data}
\alias{regression.surrogate_aug_data}
\title{Augment the training data and the explicands}
\usage{
regression.surrogate_aug_data(
  internal,
  x,
  y_hat = NULL,
  index_features = NULL,
  augment_masks_as_factor = FALSE,
  augment_include_grand = FALSE,
  augment_add_id_coal = FALSE,
  augment_comb_prob = NULL,
  augment_weights = NULL
)
}
\arguments{
\item{internal}{List.
Holds all parameters, data, functions and computed objects used within \code{\link[=explain]{explain()}}
The list contains one or more of the elements \code{parameters}, \code{data}, \code{objects}, \code{iter_list}, \code{timing_list},
\code{main_timing_list}, \code{output}, and \code{iter_timing_list}.}

\item{x}{Data.table containing the training data.}

\item{y_hat}{Vector of numerics (optional) containing the predicted responses for the observations in \code{x}.}

\item{index_features}{Array of integers (optional) containing which coalitions to consider. Must be provided if
\code{x} is the explicands.}

\item{augment_masks_as_factor}{Logical (default is \code{FALSE}). If \code{TRUE}, then the binary masks are converted
to factors. If \code{FALSE}, then the binary masks are numerics.}

\item{augment_include_grand}{Logical (default is \code{FALSE}). If \code{TRUE}, then the grand coalition is included.
If \code{index_features} are provided, then \code{augment_include_grand} has no effect. Note that if we sample the
coalitions then the grand coalition is equally likely to be samples as the other coalitions (or weighted if
\code{augment_comb_prob} is provided).}

\item{augment_add_id_coal}{Logical (default is \code{FALSE}). If \code{TRUE}, an additional column is adding containing
which coalition was applied.}

\item{augment_comb_prob}{Array of numerics (default is \code{NULL}). The length of the array must match the number of
coalitions being considered, where each entry specifies the probability of sampling the corresponding coalition.
This is useful if we want to generate more training data for some specific coalitions. One possible choice would be
\code{augment_comb_prob = if (use_Shapley_weights) internal$objects$X$shapley_weight[2:actual_n_coalitions] else NULL}.}

\item{augment_weights}{String (optional). Specifying which type of weights to add to the observations.
If \code{NULL} (default), then no weights are added. If \code{"Shapley"}, then the Shapley weights for the different
coalitions are added to corresponding observations where the coalitions was applied. If \code{uniform}, then
all observations get an equal weight of one.}
}
\value{
A data.table containing the augmented data.
}
\description{
Augment the training data and the explicands
}
\author{
Lars Henry Berge Olsen
}
\keyword{internal}
