% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/cli.R
\name{cli_topline}
\alias{cli_topline}
\title{Create a header topline with cli}
\usage{
cli_topline(verbose, testing, init_time, type, is_python)
}
\arguments{
\item{verbose}{String vector or NULL.
Specifies the verbosity (printout detail level) through one or more of strings \code{"basic"}, \code{"progress"},
\code{"convergence"}, \code{"shapley"} and \code{"vS_details"}.
\code{"basic"} (default) displays basic information about the computation which is being performed,
in addition to some messages about parameters being sets or checks being unavailable due to specific input.
\verb{"progress} displays information about where in the calculation process the function currently is.
#' \code{"convergence"} displays information on how close to convergence the Shapley value estimates are
(only when \code{iterative = TRUE}) .
\code{"shapley"} displays intermediate Shapley value estimates and standard deviations (only when \code{iterative = TRUE})
and the final estimates.
\code{"vS_details"} displays information about the v_S estimates.
This is most relevant for \verb{approach \%in\% c("regression_separate", "regression_surrogate", "vaeac"}).
\code{NULL} means no printout.
Note that any combination of four strings can be used.
E.g. \code{verbose = c("basic", "vS_details")} will display basic information + details about the v(S)-estimation process.}

\item{testing}{Logical.
Only use to remove random components like timing from the object output when comparing output with testthat.
Defaults to \code{FALSE}.}

\item{init_time}{POSIXct object.
The time when the \code{explain()} function was called, as outputted by \code{Sys.time()}.
Used to calculate the time it took to run the full \code{explain} call.}

\item{type}{Character.
Either "regular" or "forecast" corresponding to function \code{setup()} is called from,
correspondingly the type of explanation that should be generated.}

\item{is_python}{Logical.
Indicates whether the function is called from the Python wrapper.
Default is FALSE which is never changed when calling the function via \code{explain()} in R.
The parameter is later used to disallow running the AICc-versions of the empirical method
as that requires data based optimization, which is not supported in \code{shaprpy}.}
}
\value{
No return value (but prints header with cli unless \code{verbose} is \code{NULL})
}
\description{
Create a header topline with cli
}
\keyword{internal}
