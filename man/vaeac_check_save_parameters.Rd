% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/approach_vaeac.R
\name{vaeac_check_save_parameters}
\alias{vaeac_check_save_parameters}
\title{Function that gives a warning about disk usage}
\usage{
vaeac_check_save_parameters(
  save_data,
  epochs,
  save_every_nth_epoch,
  x_train_size
)
}
\arguments{
\item{save_data}{Logical (default is \code{FALSE}). If \code{TRUE}, then the data is stored together with
the model. Useful if one are to continue to train the model later using \code{\link[=vaeac_train_model_continue]{vaeac_train_model_continue()}}.}

\item{epochs}{Positive integer (default is \code{100}). The number of epochs to train the final vaeac model.
This includes \code{epochs_initiation_phase}, where the default is \code{2}.}

\item{save_every_nth_epoch}{Positive integer (default is \code{NULL}). If provided, then the vaeac model after
every \code{save_every_nth_epoch}th epoch will be saved.}

\item{x_train_size}{The object size of the \code{x_train} object.}
}
\value{
The function does not return anything.
}
\description{
Function that gives a warning about disk usage
}
\author{
Lars Henry Berge Olsen
}
\keyword{internal}
