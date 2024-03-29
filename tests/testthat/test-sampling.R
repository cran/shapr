context("test-sample_combinations.R")

test_that("Test sample_combinations", {

  # Example -----------
  ntrain <- 10
  ntest <- 10
  nsamples <- 7
  joint_sampling <- FALSE
  cnms <- c("samp_train", "samp_test")

  set.seed(123) # Ensuring consistency in every test
  x <- sample_combinations(ntrain, ntest, nsamples, joint_sampling)

  # Tests -----------
  expect_true(is.data.frame(x))
  expect_equal(names(x), cnms)
  expect_equal(nrow(x), nsamples)

  # Expect all unique values when nsamples < ntrain
  expect_true(length(unique(x$samp_train)) == nsamples)
  expect_true(length(unique(x$samp_test)) == nsamples)

  expect_true(max(x$samp_train) <= ntrain)
  expect_true(max(x$samp_test) <= ntest)

  # Example -----------
  ntrain <- 5
  ntest <- 5
  nsamples <- 7
  joint_sampling <- FALSE

  x <- sample_combinations(ntrain, ntest, nsamples, joint_sampling)

  # Tests -----------
  expect_true(max(x$samp_train) <= ntrain)
  expect_true(max(x$samp_test) <= ntest)
  expect_equal(nrow(x), nsamples)

  # Example -----------
  ntrain <- 5
  ntest <- 5
  nsamples <- 7
  joint_sampling <- TRUE

  x <- sample_combinations(ntrain, ntest, nsamples, joint_sampling)

  # Tests -----------
  expect_true(max(x$samp_train) <= ntrain)
  expect_true(max(x$samp_test) <= ntest)
  expect_equal(nrow(x), nsamples)
})

test_that("test sample_gaussian", {
  if (requireNamespace("MASS", quietly = TRUE)) {
    # Example -----------
    m <- 10
    n_samples <- 50
    mu <- rep(1, m)
    cov_mat <- cov(matrix(rnorm(n_samples * m), n_samples, m))
    x_test <- matrix(MASS::mvrnorm(1, mu, cov_mat), nrow = 1)
    cnms <- paste0("x", seq(m))
    colnames(x_test) <- cnms
    index_given <- c(4, 7)
    r <- sample_gaussian(index_given, n_samples, mu, cov_mat, m, x_test)

    # Test output format ------------------
    expect_true(data.table::is.data.table(r))
    expect_equal(ncol(r), m)
    expect_equal(nrow(r), n_samples)
    expect_equal(colnames(r), cnms)

    # Check that the given features are not resampled, but kept as is.
    for (i in seq(m)) {
      var_name <- cnms[i]

      if (i %in% index_given) {
        expect_equal(
          unique(r[[var_name]]), x_test[, var_name][[1]]
        )
      } else {
        expect_true(
          length(unique(r[[var_name]])) == n_samples
        )
      }
    }

    # Example 2 -------------
    # Check that conditioning upon all variables simply returns the test observation.
    r <- sample_gaussian(1:m, n_samples, mu, cov_mat, m, x_test)
    expect_identical(r, data.table::as.data.table(x_test))

    # Tests for errors ------------------
    expect_error(
      sample_gaussian(m + 1, n_samples, mu, cov_mat, m, x_test)
    )
    expect_error(
      sample_gaussian(m + 1, n_samples, mu, cov_mat, m, as.vector(x_test))
    )
  }
})

test_that("test sample_copula", {
  if (requireNamespace("MASS", quietly = TRUE)) {
    # Example 1 --------------
    # Check that the given features are not resampled, but kept as is.
    m <- 10
    n <- 40
    n_samples <- 50
    mu <- rep(1, m)
    set.seed(123) # Ensuring consistency in every test
    cov_mat <- cov(matrix(rnorm(n * m), n, m))
    x_train <- MASS::mvrnorm(n, mu, cov_mat)
    x_test <- MASS::mvrnorm(1, mu, cov_mat)
    x_test_gaussian <- MASS::mvrnorm(1, mu, cov_mat)
    index_given <- 3:6
    set.seed(1)
    ret <- sample_copula(index_given, n_samples, mu, cov_mat, m, x_test_gaussian, x_train, x_test)
    X_given <- x_test[index_given]
    res1.1 <- as.data.table(matrix(rep(X_given, each = n_samples), nrow = n_samples))
    res1.2 <- as.data.table(ret[, ..index_given])
    colnames(res1.1) <- colnames(res1.2)

    # Example 2 --------------
    # Check that conditioning upon all variables simply returns the test observation.
    index_given <- 1:m
    x2 <- as.data.table(matrix(x_test, ncol = m, nrow = 1))
    res2 <- sample_copula(index_given, n_samples, mu, cov_mat, m, x_test_gaussian, x_train, x_test)

    # Example 3 --------------
    # Check that the colnames are preserved.
    index_given <- c(1, 2, 3, 5, 6)
    x_test <- t(as.data.frame(x_test))
    colnames(x_test) <- 1:m
    res3 <- sample_copula(index_given, n_samples, mu, cov_mat, m, x_test_gaussian, x_train, x_test)

    # Tests ------------------
    expect_equal(res1.1, res1.2)
    expect_equal(x2, res2)
    expect_identical(colnames(res3), colnames(x_test))
    expect_error(sample_copula(m + 1, n_samples, mu, cov_mat, m, x_test_gaussian, x_train, x_test))
    expect_true(data.table::is.data.table(res2))
  }
})

test_that("test create_ctree", {
  if (requireNamespace("MASS", quietly = TRUE) & requireNamespace("party", quietly = TRUE)) {

    # Example 1-----------
    m <- 10
    n <- 40
    n_samples <- 50
    mu <- rep(1, m)
    set.seed(123) # Ensuring consistency in every test
    cov_mat <- cov(matrix(rnorm(n * m), n, m))
    x_train <- data.table::data.table(MASS::mvrnorm(n, mu, cov_mat))

    given_ind <- c(4, 7)

    mincriterion <- 0.95
    minsplit <- 20
    minbucket <- 7
    sample <- TRUE

    # build the tree
    r <- create_ctree(
      given_ind = given_ind,
      x_train = x_train,
      mincriterion = mincriterion,
      minsplit = minsplit,
      minbucket = minbucket,
      use_partykit = "on_error"
    )

    dependent_ind <- (1:dim(x_train)[2])[-given_ind]
    # Test output format ------------------
    expect_true(is.list(r))
    expect_equal(length(r), 3)
    expect_equal(class(r$tree)[1], "BinaryTree")
    expect_equal(r$given_ind, given_ind)
    expect_equal(r$dependent_ind, dependent_ind)

    df <- data.table(cbind(
      party::response(object = r$tree)$Y1,
      party::response(object = r$tree)$Y2,
      party::response(object = r$tree)$Y3,
      party::response(object = r$tree)$Y4,
      party::response(object = r$tree)$Y5,
      party::response(object = r$tree)$Y6,
      party::response(object = r$tree)$Y7,
      party::response(object = r$tree)$Y8
    ))

    names(df) <- paste0("V", dependent_ind)
    expect_equal(df, x_train[, dependent_ind, with = FALSE])

    # Example 2 -------------
    # Check that conditioning upon all variables returns empty tree.

    given_ind <- 1:10
    mincriterion <- 0.95
    minsplit <- 20
    minbucket <- 7
    sample <- TRUE

    # build the tree
    r <- create_ctree(
      given_ind = given_ind,
      x_train = x_train,
      mincriterion = mincriterion,
      minsplit = minsplit,
      minbucket = minbucket,
      use_partykit = "on_error"
    )

    expect_equal(length(r), 3)
    expect_true(is.list(r))
    expect_true(is.list(r$tree))
    expect_equal(r$given_ind, given_ind)
    expect_equal(r$dependent_ind, (1:dim(x_train)[2])[-given_ind])
  }
})

test_that("test sample_ctree", {
  if (requireNamespace("MASS", quietly = TRUE) & requireNamespace("party", quietly = TRUE)) {
    # Example -----------
    m <- 10
    n <- 40
    n_samples <- 50
    mu <- rep(1, m)
    set.seed(123) # Ensuring consistency in every test
    cov_mat <- cov(matrix(rnorm(n * m), n, m))
    x_train <- data.table::data.table(MASS::mvrnorm(n, mu, cov_mat))
    x_test <- MASS::mvrnorm(1, mu, cov_mat)
    x_test_dt <- data.table::setDT(as.list(x_test))

    given_ind <- c(4, 7)

    # build the tree
    dependent_ind <- (1:dim(x_train)[2])[-given_ind]

    x <- x_train[, given_ind, with = FALSE]
    y <- x_train[, dependent_ind, with = FALSE]

    df <- data.table::data.table(cbind(y, x))

    colnames(df) <- c(paste0("Y", 1:ncol(y)), paste0("V", given_ind))

    ynam <- paste0("Y", 1:ncol(y))
    fmla <- as.formula(paste(paste(ynam, collapse = "+"), "~ ."))

    datact <- party::ctree(fmla,
      data = df, controls =
        party::ctree_control(
          minbucket = 7,
          mincriterion = 0.95
        )
    )


    tree <- list(tree = datact, given_ind = given_ind, dependent_ind = dependent_ind)

    # new
    r <- sample_ctree(
      tree = tree, n_samples = n_samples, x_test = x_test_dt,
      x_train = x_train,
      p = length(x_test), sample = TRUE
    )

    # Test output format ------------------
    expect_true(data.table::is.data.table(r))
    expect_equal(ncol(r), m)
    expect_equal(nrow(r), n_samples)
    expect_equal(colnames(r), colnames(x_test_dt))

    # Example 2 -------------
    # Check that conditioning upon all variables simply returns the test observation.

    given_ind <- 1:10
    dependent_ind <- (1:dim(x_train)[2])[-given_ind]
    datact <- list()
    tree <- list(tree = datact, given_ind = given_ind, dependent_ind = dependent_ind)
    r <- sample_ctree(
      tree = tree, n_samples = n_samples, x_test = x_test_dt,
      x_train = x_train,
      p = length(x_test), sample = TRUE
    )
    expect_identical(r, data.table::as.data.table(x_test_dt))
  }
})
