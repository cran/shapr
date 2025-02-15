% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/approach_vaeac.R
\name{vaeac_check_epoch_values}
\alias{vaeac_check_epoch_values}
\title{Function that checks provided epoch arguments}
\usage{
vaeac_check_epoch_values(
  epochs,
  epochs_initiation_phase,
  epochs_early_stopping,
  save_every_nth_epoch
)
}
\arguments{
\item{epochs}{Positive integer (default is \code{100}). The number of epochs to train the final vaeac model.
This includes \code{epochs_initiation_phase}, where the default is \code{2}.}

\item{epochs_initiation_phase}{Positive integer (default is \code{2}). The number of epochs to run each of the
\code{n_vaeacs_initialize} \code{vaeac} models before continuing to train only the best performing model.}

\item{epochs_early_stopping}{Positive integer (default is \code{NULL}). The training stops if there has been no
improvement in the validation IWAE for \code{epochs_early_stopping} epochs. If the user wants the training process
to be solely based on this training criterion, then \code{epochs} in \code{\link[=explain]{explain()}} should be set to a large
number. If \code{NULL}, then \code{shapr} will internally set \code{epochs_early_stopping = vaeac.epochs} such that early
stopping does not occur.}

\item{save_every_nth_epoch}{Positive integer (default is \code{NULL}). If provided, then the vaeac model after
every \code{save_every_nth_epoch}th epoch will be saved.}
}
\value{
The function does not return anything.
}
\description{
Function that checks provided epoch arguments
}
\author{
Lars Henry Berge Olsen
}
\keyword{internal}
