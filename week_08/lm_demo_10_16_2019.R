### simple lm example with 2 "real" inputs and 10 "fake" or
### "inactive" inputs

library(dplyr)
library(ggplot2)

### create the toy demo input values
### define a mean vector and covariance matrix
mu_x <- c(-0.5, 0.25)
sigma_1 <- 1.25
sigma_2 <- 1.5
rho_x <- 0.25
covmat_x <- matrix(c(sigma_1^2,
                     rho_x*sigma_1*sigma_2,
                     rho_x*sigma_1*sigma_2,
                     sigma_2^2),
                   nrow = 2,
                   byrow = 2)

covmat_x

### generate random observations of the inputs
set.seed(12345)
Xreal <- MASS::mvrnorm(n = 250,
                       mu = mu_x,
                       Sigma = covmat_x) %>% 
  as.data.frame() %>% tbl_df()

### create "fake" inputs
Xfake <- MASS::mvrnorm(n = 250,
                       mu = rep(0, 10),
                       Sigma = diag(10)) %>% 
  as.data.frame() %>% tbl_df()

### combine the real and fake inputs together 
Xall <- Xreal %>% 
  bind_cols(Xfake) %>% 
  purrr::set_names(sprintf("x%02d", 1:(ncol(Xreal)+ncol(Xfake))))

### define a true function
my_true_function <- function(x1, x2, b_vec)
{
  b_vec[1] + b_vec[2] * x1 + b_vec[3] * x2 + b_vec[4] * x1 * x2
}

### create the full dataset of real and fake inputs and the
### true linear predictor
beta_true <- c(0.75, -1.2, 1.5, 2.5)

true_dataset <- Xall %>% 
  mutate(mu = my_true_function(x01, x02, beta_true))

### visualize the linear predictor wrt the true inputs
true_dataset %>% 
  select(mu, x01, x02) %>% 
  tibble::rowid_to_column("obs_id") %>% 
  tidyr::gather(key = "input_name", 
                value = "input_value", 
                -obs_id,
                -mu) %>% 
  ggplot(mapping = aes(x = input_value,
                       y = mu)) +
  geom_point() +
  facet_grid(~input_name)

### now visualize the trends of the true linear predictor
### with respect to all 12 inputs, including the fake inputs
true_dataset %>% 
  tibble::rowid_to_column("obs_id") %>% 
  tidyr::gather(key = "input_name",
                value = "input_value",
                -obs_id,
                -mu) %>% 
  ggplot(mapping = aes(x = input_value,
                       y = mu)) +
  geom_point() +
  facet_wrap(~input_name)

### set the noise
sigma_true <- 1

### generate the random noise
set.seed(34114)
noisy_dataset <- true_dataset %>% 
  mutate(y = rnorm(n = n(),
                   mean = mu,
                   sd = sigma_true))

### create the training and holdout datasets with a data split
set.seed(389143)
train_id <- sample(1:nrow(noisy_dataset), 100)

train_df <- noisy_dataset %>% 
  slice(train_id)

holdout_df <- noisy_dataset %>% 
  slice(-train_id)

### visualize the noisy response with respect to each input
train_df %>% 
  select(-mu) %>% 
  tibble::rowid_to_column("obs_id") %>% 
  tidyr::gather(key = "input_name",
                value = "input_value",
                -obs_id,
                -y) %>% 
  ggplot(mapping = aes(x = input_value,
                       y = y)) +
  geom_point() +
  facet_wrap(~input_name)

### now let's build some simple non-bayesian linear models
### with the `lm()` function

### x01 and x02 with two fake inputs
mod1 <- lm(y ~ x01 + x02 + x03 + x04, train_df)

summary(mod1)

### visualize the parameter (coefficients) with `coefplot`
# install.packages("coefplot")

coefplot::coefplot(mod1)

### the interaction model of just the real inputs
mod2 <- lm(y ~ x01*x02, train_df)

coefplot::coefplot(mod2)
