% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/approach_vaeac.R
\name{vaeac_get_full_state_list}
\alias{vaeac_get_full_state_list}
\title{Function that extracts the state list objects from the environment}
\usage{
vaeac_get_full_state_list(environment)
}
\arguments{
\item{environment}{The \code{\link[base:environment]{base::environment()}} where the objects are stored.}
}
\value{
List containing the values of \code{norm_mean}, \code{norm_std}, \code{model_description}, \code{folder_to_save_model},
\code{n_train}, \code{n_features}, \code{one_hot_max_sizes}, \code{epochs}, \code{epochs_specified}, \code{epochs_early_stopping},
\code{early_stopping_applied}, \code{running_avg_n_values}, \code{paired_sampling}, \code{mask_generator_name}, \code{masking_ratio},
\code{mask_gen_coalitions}, \code{mask_gen_coalitions_prob}, \code{val_ratio}, \code{val_iwae_n_samples},
\code{n_vaeacs_initialize}, \code{epochs_initiation_phase}, \code{width}, \code{depth}, \code{latent_dim}, \code{activation_function},
\code{lr}, \code{batch_size}, \code{skip_conn_layer}, \code{skip_conn_masked_enc_dec}, \code{batch_normalization}, \code{cuda},
\code{train_indices}, \code{val_indices}, \code{save_every_nth_epoch}, \code{sigma_mu},
\code{sigma_sigma}, \code{feature_list}, \code{col_cat_names}, \code{col_cont_names}, \code{col_cat}, \code{col_cont}, \code{cat_in_dataset},
\code{map_new_to_original_names}, \code{map_original_to_new_names}, \code{log_exp_cont_feat}, \code{save_data}, \code{verbose},
\code{seed}, and \code{vaeac_save_file_names}.
}
\description{
#' @description
The function extract the objects that we are going to save together with the \code{vaeac} model to make it possible to
train the model further and to evaluate it.
The environment should be the local environment inside the \code{\link[=vaeac_train_model_auxiliary]{vaeac_train_model_auxiliary()}} function.
}
\author{
Lars Henry Berge Olsen
}
\keyword{internal}
