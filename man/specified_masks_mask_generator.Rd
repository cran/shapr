% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/approach_vaeac_torch_modules.R
\name{specified_masks_mask_generator}
\alias{specified_masks_mask_generator}
\title{A \code{\link[torch:nn_module]{torch::nn_module()}} Representing a specified_masks_mask_generator}
\usage{
specified_masks_mask_generator(masks, masks_probs, paired_sampling = FALSE)
}
\arguments{
\item{masks}{Matrix/Tensor of possible/allowed 'masks' which we sample from.}

\item{masks_probs}{Array of 'probabilities' for each of the masks specified in 'masks'.
Note that they do not need to be between 0 and 1 (e.g. sampling frequency).
They are scaled, hence, they only need to be positive.}

\item{paired_sampling}{Boolean. If we are doing paired sampling. So include both S and \eqn{\bar{S}}.
If TRUE, then batch must be sampled using 'paired_sampler' which creates batches where
the first half and second half of the rows are duplicates of each other. That is,
\verb{batch = [row1, row1, row2, row2, row3, row3, ...]}.}
}
\description{
A mask generator which masks the entries based on sampling provided 1D masks with corresponding probabilities.
Used for Shapley value estimation when only a subset of coalitions are used to compute the Shapley values.
}
\author{
Lars Henry Berge Olsen
}
\keyword{internal}
