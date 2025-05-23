% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/plot.R
\name{process_factor_data}
\alias{process_factor_data}
\title{Treat factors as numeric values}
\usage{
process_factor_data(dt, factor_cols)
}
\arguments{
\item{dt}{data.table to plot}

\item{factor_cols}{Columns that are factors or character}
}
\value{
A list of a lookup table with each factor and level and its numeric value, a data.table
very similar to the input data, but now with numeric values for factors, and the maximum feature value.
}
\description{
Factors are given a numeric value above the highest numeric value in the data. The value of the different levels
are sorted by factor and then level.
}
\keyword{internal}
