% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/preprocess_data.R
\name{get_data_specs}
\alias{get_data_specs}
\title{Fetches feature information from a given data set}
\usage{
get_data_specs(x)
}
\arguments{
\item{x}{matrix, data.frame or data.table The data to extract feature information from.}
}
\value{
A list with the following elements:
\describe{
  \item{labels}{character vector with the feature names to compute Shapley values for}
  \item{classes}{a named character vector with the labels as names and the class types as elements}
  \item{factor_levels}{a named list with the labels as names and character vectors with the factor levels as elements
  (NULL if the feature is not a factor)}
}
}
\description{
Fetches feature information from a given data set
}
\details{
This function is used to extract the feature information to be checked against the corresponding
information extracted from the model and other data sets. The function is called from
\code{\link[shapr:preprocess_data]{preprocess_data}}
and \code{\link[shapr:make_dummies]{make_dummies}}
}
\examples{
# Load example data
if (requireNamespace("MASS", quietly = TRUE)) {
  data("Boston", package = "MASS")
  # Split data into test- and training data
  x_train <- data.table::as.data.table(head(Boston))
  x_train[, rad := as.factor(rad)]
  get_data_specs(x_train)
}
}
\author{
Martin Jullum
}
\keyword{internal}
