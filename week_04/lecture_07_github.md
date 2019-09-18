INFSCI 2595: Lecture 07
================
Dr. Joseph P. Yurko
September 18, 2019

Overview
--------

This week, the lecture material will be entirely contained within the R Markdown rendered report.

Load packages
-------------

This document uses the following packages:

``` r
library(dplyr)
library(ggplot2)
```

In addition, functions from the following `tidyverse` packages are used: `tibble`, `tidyr`, and `purrr`. To completely run all code chunks within the markdown, you must also have the `mvtnorm` package downloaded and installed. The `MASS` package is also required to complete all code chunks.

Model formulation
-----------------

Rather than discussing linear models in terms of fitting a relationship to observed data, let's start out by discussing the model formulation. We wish to model the relationship between a response, output, or outcome, ![y](https://latex.codecogs.com/png.latex?y "y"), with respect to an input ![x](https://latex.codecogs.com/png.latex?x "x"). Further, we will assume, for now, that we can use a linear relationship to relate the input to the response. If we observed ![n = 1,...,N](https://latex.codecogs.com/png.latex?n%20%3D%201%2C...%2CN "n = 1,...,N") observations, the model is typically written as:

![ 
y\_{n} = \\beta\_{0} + \\beta\_{1}x\_n + \\epsilon\_n \\\\ \\epsilon\_n \\sim \\mathrm{normal}\\left(0, \\sigma\\right)
](https://latex.codecogs.com/png.latex?%20%0Ay_%7Bn%7D%20%3D%20%5Cbeta_%7B0%7D%20%2B%20%5Cbeta_%7B1%7Dx_n%20%2B%20%5Cepsilon_n%20%5C%5C%20%5Cepsilon_n%20%5Csim%20%5Cmathrm%7Bnormal%7D%5Cleft%280%2C%20%5Csigma%5Cright%29%0A " 
y_{n} = \beta_{0} + \beta_{1}x_n + \epsilon_n \\ \epsilon_n \sim \mathrm{normal}\left(0, \sigma\right)
")

However, in this course, we will not use that notation. Instead of relegating the Gaussian distribution to "the end", appearing in the ![\\epsilon](https://latex.codecogs.com/png.latex?%5Cepsilon "\epsilon") term, we will bring the Gaussian distribution to the forefront and define our model as:

![ 
y\_n \\mid x\_{n},\\boldsymbol{\\beta},\\sigma \\sim \\mathrm{normal}\\left( \\beta\_{0} + \\beta\_{1}x\_n, \\sigma \\right)
](https://latex.codecogs.com/png.latex?%20%0Ay_n%20%5Cmid%20x_%7Bn%7D%2C%5Cboldsymbol%7B%5Cbeta%7D%2C%5Csigma%20%5Csim%20%5Cmathrm%7Bnormal%7D%5Cleft%28%20%5Cbeta_%7B0%7D%20%2B%20%5Cbeta_%7B1%7Dx_n%2C%20%5Csigma%20%5Cright%29%0A " 
y_n \mid x_{n},\boldsymbol{\beta},\sigma \sim \mathrm{normal}\left( \beta_{0} + \beta_{1}x_n, \sigma \right)
")

At first this might seem like an odd change to make, but it highlights several important aspects of the linear model. First, the observed response, ![y\_{n}](https://latex.codecogs.com/png.latex?y_%7Bn%7D "y_{n}"), is normally distributed **given** the input, ![x\_{n}](https://latex.codecogs.com/png.latex?x_%7Bn%7D "x_{n}"), and the parameters, ![\\beta\_{0}](https://latex.codecogs.com/png.latex?%5Cbeta_%7B0%7D "\beta_{0}"), ![\\beta\_{1}](https://latex.codecogs.com/png.latex?%5Cbeta_%7B1%7D "\beta_{1}"), and ![\\sigma](https://latex.codecogs.com/png.latex?%5Csigma "\sigma"). Second, based on the notation we have used through this course, the *mean* of the ![y\_{n}](https://latex.codecogs.com/png.latex?y_%7Bn%7D "y_{n}") data point is the functional relationship, ![\\beta\_{0} + \\beta\_{1}x\_{n}](https://latex.codecogs.com/png.latex?%5Cbeta_%7B0%7D%20%2B%20%5Cbeta_%7B1%7Dx_%7Bn%7D "\beta_{0} + \beta_{1}x_{n}"). We will now make this point even more clear by rewriting our model in terms of the *linear predictor* ![\\mu\_{n}](https://latex.codecogs.com/png.latex?%5Cmu_%7Bn%7D "\mu_{n}"):

![ 
y\_{n} \\mid \\mu\_{n},\\sigma \\sim \\mathrm{normal}\\left( \\mu\_{n}, \\sigma \\right) \\\\ \\mu\_{n} = \\beta\_{0} + \\beta\_{1}x\_n
](https://latex.codecogs.com/png.latex?%20%0Ay_%7Bn%7D%20%5Cmid%20%5Cmu_%7Bn%7D%2C%5Csigma%20%5Csim%20%5Cmathrm%7Bnormal%7D%5Cleft%28%20%5Cmu_%7Bn%7D%2C%20%5Csigma%20%5Cright%29%20%5C%5C%20%5Cmu_%7Bn%7D%20%3D%20%5Cbeta_%7B0%7D%20%2B%20%5Cbeta_%7B1%7Dx_n%0A " 
y_{n} \mid \mu_{n},\sigma \sim \mathrm{normal}\left( \mu_{n}, \sigma \right) \\ \mu_{n} = \beta_{0} + \beta_{1}x_n
")

The first line in the above equation block, is the *stochastic*, or random, portion of the model. It says that the response ![y\_{n}](https://latex.codecogs.com/png.latex?y_%7Bn%7D "y_{n}") is normally distributed given a *mean*, ![\\mu\_{n}](https://latex.codecogs.com/png.latex?%5Cmu_%7Bn%7D "\mu_{n}"), and standard deviation, ![\\sigma](https://latex.codecogs.com/png.latex?%5Csigma "\sigma"). This is the same setup as the normal likelihood we used in Lecture 04...**except**...the mean is not a constant. The mean, ![\\mu\_{n}](https://latex.codecogs.com/png.latex?%5Cmu_%7Bn%7D "\mu_{n}"), can change observation to observation. Thus, the ![\\left( n-5 \\right)](https://latex.codecogs.com/png.latex?%5Cleft%28%20n-5%20%5Cright%29 "\left( n-5 \right)")-th observation may have a different mean than the ![\\left(n + 5\\right)](https://latex.codecogs.com/png.latex?%5Cleft%28n%20%2B%205%5Cright%29 "\left(n + 5\right)")-th observation.

How does the mean vary? It is function of the input ![x](https://latex.codecogs.com/png.latex?x "x") and two additional parameters ![\\beta\_{0}](https://latex.codecogs.com/png.latex?%5Cbeta_%7B0%7D "\beta_{0}") and ![\\beta\_{1}](https://latex.codecogs.com/png.latex?%5Cbeta_%7B1%7D "\beta_{1}"). The vector ![\\boldsymbol{\\beta}](https://latex.codecogs.com/png.latex?%5Cboldsymbol%7B%5Cbeta%7D "\boldsymbol{\beta}") will be used to denote all of the *linear predictor* parameters. Notice that the mean, ![\\mu\_{n}](https://latex.codecogs.com/png.latex?%5Cmu_%7Bn%7D "\mu_{n}"), is defined with an equal sign above. This is to denote that the mean is a **deterministic** function. If we know the ![\\boldsymbol{\\beta}](https://latex.codecogs.com/png.latex?%5Cboldsymbol%7B%5Cbeta%7D "\boldsymbol{\beta}") parameters and the input ![x\_{n}](https://latex.codecogs.com/png.latex?x_%7Bn%7D "x_{n}"), then we will know ![\\mu\_{n}](https://latex.codecogs.com/png.latex?%5Cmu_%7Bn%7D "\mu_{n}"), precisely. The phrase *linear predictor* represents that ![\\mu\_{n}](https://latex.codecogs.com/png.latex?%5Cmu_%7Bn%7D "\mu_{n}"), is the "prediction" from the linear model.

The parameters within the expression for the *linear predictor*, ![\\beta\_{0}](https://latex.codecogs.com/png.latex?%5Cbeta_%7B0%7D "\beta_{0}") and ![\\beta\_{1}](https://latex.codecogs.com/png.latex?%5Cbeta_%7B1%7D "\beta_{1}"), correspond to the intercept and the slope, respectively. The slope, ![\\beta\_{1}](https://latex.codecogs.com/png.latex?%5Cbeta_%7B1%7D "\beta_{1}"), controls the change in ![\\mu](https://latex.codecogs.com/png.latex?%5Cmu "\mu") with respect to a change in the input ![x](https://latex.codecogs.com/png.latex?x "x"). The intercept, ![\\beta\_{0}](https://latex.codecogs.com/png.latex?%5Cbeta_%7B0%7D "\beta_{0}"), is the value ![\\mu](https://latex.codecogs.com/png.latex?%5Cmu "\mu") takes when ![x = 0](https://latex.codecogs.com/png.latex?x%20%3D%200 "x = 0").

It is important to note that in the equations above, we factored the likelihood on all observed responses into the product of ![N](https://latex.codecogs.com/png.latex?N "N") separate likelihoods. Therefore, we assumed the ![\\mathbf{y} = \\{y\_1, y\_2,...,y\_N\\}](https://latex.codecogs.com/png.latex?%5Cmathbf%7By%7D%20%3D%20%5C%7By_1%2C%20y_2%2C...%2Cy_N%5C%7D "\mathbf{y} = \{y_1, y_2,...,y_N\}") responses are **conditionally** independent given the inputs and the model parameters. The equation block below makes that explicit by writing out the joint likelihood:

![ 
p \\left(\\mathbf{y} \\mid \\mathbf{x}, \\boldsymbol{\\beta}, \\sigma \\right) = \\prod\_{n=1}^{N} \\left( p\\left( y\_{n} \\mid \\mu\_{n}, \\sigma \\right) \\right) \\\\ \\mu\_{n} = \\beta\_{0} + \\beta\_{1} x\_{n}
](https://latex.codecogs.com/png.latex?%20%0Ap%20%5Cleft%28%5Cmathbf%7By%7D%20%5Cmid%20%5Cmathbf%7Bx%7D%2C%20%5Cboldsymbol%7B%5Cbeta%7D%2C%20%5Csigma%20%5Cright%29%20%3D%20%5Cprod_%7Bn%3D1%7D%5E%7BN%7D%20%5Cleft%28%20p%5Cleft%28%20y_%7Bn%7D%20%5Cmid%20%5Cmu_%7Bn%7D%2C%20%5Csigma%20%5Cright%29%20%5Cright%29%20%5C%5C%20%5Cmu_%7Bn%7D%20%3D%20%5Cbeta_%7B0%7D%20%2B%20%5Cbeta_%7B1%7D%20x_%7Bn%7D%0A " 
p \left(\mathbf{y} \mid \mathbf{x}, \boldsymbol{\beta}, \sigma \right) = \prod_{n=1}^{N} \left( p\left( y_{n} \mid \mu_{n}, \sigma \right) \right) \\ \mu_{n} = \beta_{0} + \beta_{1} x_{n}
")

Model behavior
--------------

### Linear predictor

Let's get some practice visualizing how the parameters, ![\\boldsymbol{\\beta}](https://latex.codecogs.com/png.latex?%5Cboldsymbol%7B%5Cbeta%7D "\boldsymbol{\beta}") and ![\\sigma](https://latex.codecogs.com/png.latex?%5Csigma "\sigma"), influence the relationship between the input and the response. The ![\\boldsymbol{\\beta}](https://latex.codecogs.com/png.latex?%5Cboldsymbol%7B%5Cbeta%7D "\boldsymbol{\beta}") define the relationship between the input, ![x](https://latex.codecogs.com/png.latex?x "x"), and the linear predictor ![\\mu](https://latex.codecogs.com/png.latex?%5Cmu "\mu"). ![\\beta\_{0}](https://latex.codecogs.com/png.latex?%5Cbeta_%7B0%7D "\beta_{0}") corresponds to the intercept while ![\\beta\_{1}](https://latex.codecogs.com/png.latex?%5Cbeta_%7B1%7D "\beta_{1}") is the slope. The slope, ![\\beta\_{1}](https://latex.codecogs.com/png.latex?%5Cbeta_%7B1%7D "\beta_{1}"), controls the change in ![\\mu](https://latex.codecogs.com/png.latex?%5Cmu "\mu") with respect to a change in the input ![x](https://latex.codecogs.com/png.latex?x "x"). The intercept, ![\\beta\_{0}](https://latex.codecogs.com/png.latex?%5Cbeta_%7B0%7D "\beta_{0}"), is the value ![\\mu](https://latex.codecogs.com/png.latex?%5Cmu "\mu") takes when ![x = 0](https://latex.codecogs.com/png.latex?x%20%3D%200 "x = 0"). Even though, interpreting the influence of the intercept and slope should be straightforward, we will visualize the linear predictor as a function of the input, at several combinations of ![\\beta\_{0}](https://latex.codecogs.com/png.latex?%5Cbeta_%7B0%7D "\beta_{0}") and ![\\beta\_{1}](https://latex.codecogs.com/png.latex?%5Cbeta_%7B1%7D "\beta_{1}").

The code chunk below uses the `expand.grid()` function to create a grid of 9 combinations of the two linear predictor parameters.

``` r
beta_grid <- expand.grid(beta_0 = -1:1,
                         beta_1 = -1:1,
                         stringsAsFactors = FALSE,
                         KEEP.OUT.ATTRS = FALSE) %>% 
  as.data.frame() %>% tbl_df()
```

Next, let's define a function to calculate the linear predictor given the ![\\boldsymbol{\\beta}](https://latex.codecogs.com/png.latex?%5Cboldsymbol%7B%5Cbeta%7D "\boldsymbol{\beta}") parameters and the input ![x](https://latex.codecogs.com/png.latex?x "x").

``` r
calc_lin_predictor <- function(b0, b1, xn)
{
  
  tibble::tibble(
    x = xn,
    mu = b0 + b1 * xn
  ) %>%
    mutate(beta_0 = b0,
           beta_1 = b1)
}
```

We will evaluate the linear predictor for each of the ![\\beta\_{0}](https://latex.codecogs.com/png.latex?%5Cbeta_%7B0%7D "\beta_{0}") and ![\\beta\_{1}](https://latex.codecogs.com/png.latex?%5Cbeta_%7B1%7D "\beta_{1}") combinations over a grid of input values ![x \\in \\left\[-3, +3\\right\]](https://latex.codecogs.com/png.latex?x%20%5Cin%20%5Cleft%5B-3%2C%20%2B3%5Cright%5D "x \in \left[-3, +3\right]"). The code chunk below creates the `x` vector between those specified bounds, and the iterates over the pairs of ![\\boldsymbol{\\beta}](https://latex.codecogs.com/png.latex?%5Cboldsymbol%7B%5Cbeta%7D "\boldsymbol{\beta}") values with the `purrr` package.

``` r
x <- seq(-3, 3, length.out = 201)

lin_pred_grid <- purrr::map2_dfr(beta_grid$beta_0,
                                 beta_grid$beta_1,
                                 calc_lin_predictor,
                                 xn = x)
```

The output of the code chunk below, shows that `lin_pred_grid` is organized in a long or tall format. One row is one evaluation of the linear predictor for a particular `x` value associated with a `beta_0` and `beta_1` pair. Based on the number of rows in the `beta_grid` object, we know that there are 9 combinations of `beta_0` and `beta_1`.

``` r
lin_pred_grid %>% glimpse()
```

    ## Observations: 1,809
    ## Variables: 4
    ## $ x      <dbl> -3.00, -2.97, -2.94, -2.91, -2.88, -2.85, -2.82, -2.79,...
    ## $ mu     <dbl> 2.00, 1.97, 1.94, 1.91, 1.88, 1.85, 1.82, 1.79, 1.76, 1...
    ## $ beta_0 <int> -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,...
    ## $ beta_1 <int> -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,...

The code chunk below shows how to plot the linear predictor with respect to the input, for each of the 9 combinations. Each subplot corresponds to a specific `beta_0` and `beta_1` pair. The `beta_1` parameter varies with the horizontal subplots, while the `beta_0` parameter values with the vertical subplots. For reference a dashed horizontal line denotes ![\\mu = 0](https://latex.codecogs.com/png.latex?%5Cmu%20%3D%200 "\mu = 0") and a dashed vertical line corresponds to ![x = 0](https://latex.codecogs.com/png.latex?x%20%3D%200 "x = 0").

``` r
lin_pred_grid %>% 
  mutate(beta_0_plot = forcats::fct_rev(as.factor(beta_0))) %>% 
  ggplot(mapping = aes(x = x, y = mu)) +
  geom_hline(yintercept = 0, color = "grey50", linetype = "dashed") +
  geom_vline(xintercept = 0, color = "grey50", linetype = "dashed") +
  geom_line(mapping = aes(group = interaction(beta_0, beta_1)),
            size = 1.15) +
  facet_grid(beta_0_plot ~ beta_1,
             labeller = label_bquote(rows = beta[0]*"="*.(as.numeric(as.character(beta_0_plot))),
                                     cols = beta[1]*"="*.(beta_1))) +
  labs(y = expression(mu)) +
  theme_bw() +
  theme(strip.text = element_text(size = 12))
```

![](lecture_07_github_files/figure-markdown_github/viz_lin_pred_grid-1.png)

The trends in the above figure should make sense. Starting with the far-left column, ![\\beta\_{1} = -1](https://latex.codecogs.com/png.latex?%5Cbeta_%7B1%7D%20%3D%20-1 "\beta_{1} = -1") gives a negative relationship between ![x](https://latex.codecogs.com/png.latex?x "x") and ![\\mu](https://latex.codecogs.com/png.latex?%5Cmu "\mu"). Increasing the value of ![x](https://latex.codecogs.com/png.latex?x "x") causes ![\\mu](https://latex.codecogs.com/png.latex?%5Cmu "\mu") to decrease. Moving to the middle column, ![\\beta\_{1} = 0](https://latex.codecogs.com/png.latex?%5Cbeta_%7B1%7D%20%3D%200 "\beta_{1} = 0"), the slope is 0 and thus there is no trend at all between ![\\mu](https://latex.codecogs.com/png.latex?%5Cmu "\mu") and ![x](https://latex.codecogs.com/png.latex?x "x"). The far-right column, ![\\beta\_{1} = 1](https://latex.codecogs.com/png.latex?%5Cbeta_%7B1%7D%20%3D%201 "\beta_{1} = 1") shows a positive relationship, with increasing ![x](https://latex.codecogs.com/png.latex?x "x") resulting in an increase in ![\\mu](https://latex.codecogs.com/png.latex?%5Cmu "\mu"). Comparing the behavior between the vertical subplots shows the impact of the intercept, ![\\beta\_{0}](https://latex.codecogs.com/png.latex?%5Cbeta_%7B0%7D "\beta_{0}"). Focusing on the middle column, we see that ![\\beta\_{0}](https://latex.codecogs.com/png.latex?%5Cbeta_%7B0%7D "\beta_{0}") shifts the linear predictor up or down.

The ![\\boldsymbol{\\beta}](https://latex.codecogs.com/png.latex?%5Cboldsymbol%7B%5Cbeta%7D "\boldsymbol{\beta}") used for the previous figure were simply representative of the primary trends of a decreasing, increasing, and no-relationship between the input and the response. If we used different values for ![\\beta\_{0}](https://latex.codecogs.com/png.latex?%5Cbeta_%7B0%7D "\beta_{0}") and ![\\beta\_{1}](https://latex.codecogs.com/png.latex?%5Cbeta_%7B1%7D "\beta_{1}"), the specific plots would change, but the primary conclusions would be the same.

### Noise

Up to this point, we have not said much about the ![\\sigma](https://latex.codecogs.com/png.latex?%5Csigma "\sigma") parameter. ![\\sigma](https://latex.codecogs.com/png.latex?%5Csigma "\sigma") is the standard deviation within the likelihood of the responses given the linear predictor. Thus, ![\\sigma](https://latex.codecogs.com/png.latex?%5Csigma "\sigma") defines the variability of responses, or the *noise* of the process. The linear predictor can therefore be viewed as the *noise-free* signal "around" which the observed responses are scattered. The level or amount of scatter is controlled by ![\\sigma](https://latex.codecogs.com/png.latex?%5Csigma "\sigma"). Low values of ![\\sigma](https://latex.codecogs.com/png.latex?%5Csigma "\sigma") correspond to less noise, while higher values of ![\\sigma](https://latex.codecogs.com/png.latex?%5Csigma "\sigma") yield very noisy observations. Noise As we shall see later on, it is more difficult to learn the parameters of the linear predictor when the *noise* level is high.

To get a sense about the influence of ![\\sigma](https://latex.codecogs.com/png.latex?%5Csigma "\sigma"), let's set it at a specific value for now.

``` r
sigma_true <- 0.5
```

The previous figure is recreated below, but with ribbons included to visualize the influence of ![\\sigma](https://latex.codecogs.com/png.latex?%5Csigma "\sigma"). Three transparent ribbons are used, one between ![\\pm\\sigma](https://latex.codecogs.com/png.latex?%5Cpm%5Csigma "\pm\sigma") around the linear predictor. Another at ![\\pm2\\sigma](https://latex.codecogs.com/png.latex?%5Cpm2%5Csigma "\pm2\sigma"), and the last at ![\\pm3\\sigma](https://latex.codecogs.com/png.latex?%5Cpm3%5Csigma "\pm3\sigma") around the linear predictor.

``` r
lin_pred_grid %>% 
  mutate(beta_0_plot = forcats::fct_rev(as.factor(beta_0))) %>% 
  ggplot(mapping = aes(x = x, y = mu)) +
  geom_hline(yintercept = 0, color = "grey50", linetype = "dashed") +
  geom_vline(xintercept = 0, color = "grey50", linetype = "dashed") +
  geom_ribbon(mapping = aes(ymin = mu - 3*sigma_true,
                            ymax = mu + 3*sigma_true,
                            group = interaction(beta_0, beta_1)),
              fill = "dodgerblue", alpha = 0.25) +
  geom_ribbon(mapping = aes(ymin = mu - 2*sigma_true,
                            ymax = mu + 2*sigma_true,
                            group = interaction(beta_0, beta_1)),
              fill = "dodgerblue", alpha = 0.25) +
  geom_ribbon(mapping = aes(ymin = mu - 1*sigma_true,
                            ymax = mu + 1*sigma_true,
                            group = interaction(beta_0, beta_1)),
              fill = "dodgerblue", alpha = 0.25) +
  geom_line(mapping = aes(group = interaction(beta_0, beta_1)),
            size = 1.15) +
  facet_grid(beta_0_plot ~ beta_1,
             labeller = label_bquote(rows = beta[0]*"="*.(as.numeric(as.character(beta_0_plot))),
                                     cols = beta[1]*"="*.(beta_1))) +
  labs(y = "y") +
  theme_bw() +
  theme(strip.text = element_text(size = 12))
```

![](lecture_07_github_files/figure-markdown_github/viz_lin_pred_grid_ribbon-1.png)

It might be a little overwhelming to look at all 9 subplots. So let's go ahead and focus on a single combination of the ![\\boldsymbol{\\beta}](https://latex.codecogs.com/png.latex?%5Cboldsymbol%7B%5Cbeta%7D "\boldsymbol{\beta}") parameters. The code chunk below isolates the ![\\beta\_{1} = 1](https://latex.codecogs.com/png.latex?%5Cbeta_%7B1%7D%20%3D%201 "\beta_{1} = 1") and ![\\beta\_{0} = 1](https://latex.codecogs.com/png.latex?%5Cbeta_%7B0%7D%20%3D%201 "\beta_{0} = 1") subplot. The transparency is used to represent uncertainty. More transparency corresponds to less of a chance of an observation to occur. Remember that for a Gaussian, $$68% of the probability is contained within the ![\\pm\\sigma](https://latex.codecogs.com/png.latex?%5Cpm%5Csigma "\pm\sigma") around the mean. That is why in the figures with the ribbons, the least transparent (most opaque) portion is the inner most ribbon closest to the linear predictor line. As we move further away from the mean in either direction, the transparency increases (becomes less opaque), to denote the probability of observing a response decreases. This language shift, from linear predictor to response, is why the y-axis label changed from ![\\mu](https://latex.codecogs.com/png.latex?%5Cmu "\mu") to ![y](https://latex.codecogs.com/png.latex?y "y"). The likelihood states that specific observations are normally distributed around the linear predictor (the mean), ![\\mathrm{normal}(y\_{n} \\mid \\mu\_{n}, \\sigma)](https://latex.codecogs.com/png.latex?%5Cmathrm%7Bnormal%7D%28y_%7Bn%7D%20%5Cmid%20%5Cmu_%7Bn%7D%2C%20%5Csigma%29 "\mathrm{normal}(y_{n} \mid \mu_{n}, \sigma)"). The ribbons therefore plot the mean or expected value of the response as well as up to ![\\approx](https://latex.codecogs.com/png.latex?%5Capprox "\approx") 99.97% uncertainty interval around the linear predctor.

``` r
lin_pred_grid %>% 
  mutate(beta_0_plot = forcats::fct_rev(as.factor(beta_0))) %>% 
  filter(beta_0 == 1, beta_1 == 1) %>% 
  ggplot(mapping = aes(x = x, y = mu)) +
  geom_hline(yintercept = 0, color = "grey50", linetype = "dashed") +
  geom_vline(xintercept = 0, color = "grey50", linetype = "dashed") +
  geom_ribbon(mapping = aes(ymin = mu - 3*sigma_true,
                            ymax = mu + 3*sigma_true,
                            group = interaction(beta_0, beta_1)),
              fill = "dodgerblue", alpha = 0.25) +
  geom_ribbon(mapping = aes(ymin = mu - 2*sigma_true,
                            ymax = mu + 2*sigma_true,
                            group = interaction(beta_0, beta_1)),
              fill = "dodgerblue", alpha = 0.25) +
  geom_ribbon(mapping = aes(ymin = mu - 1*sigma_true,
                            ymax = mu + 1*sigma_true,
                            group = interaction(beta_0, beta_1)),
              fill = "dodgerblue", alpha = 0.25) +
  geom_line(mapping = aes(group = interaction(beta_0, beta_1)),
            size = 1.15) +
  facet_grid(beta_0_plot ~ beta_1,
             labeller = label_bquote(rows = beta[0]*"="*.(as.numeric(as.character(beta_0_plot))),
                                     cols = beta[1]*"="*.(beta_1))) +
  labs(y = "y") +
  theme_bw() +
  theme(strip.text = element_text(size = 12))
```

![](lecture_07_github_files/figure-markdown_github/viz_lin_pred_grid_ribbon_1-1.png)

We will now generate random draws from the likelihood, to try and make it clear that the uncertainty intervals are associated with the potential observed values. We will work with a smaller grid of ![x](https://latex.codecogs.com/png.latex?x "x") values compared with that used to draw the linear predictor. Our `calc_lin_predictor()` function is applied by a call to `purrr::map2_dfr()` to handle the book keeping associated with iterating over the ![\\boldsymbol{\\beta}](https://latex.codecogs.com/png.latex?%5Cboldsymbol%7B%5Cbeta%7D "\boldsymbol{\beta}") parameters and a small grid of 9 locations between -2 and +2. The result is piped into `mutate()` to create the variable `y`, as shown in the code chunk below.

``` r
set.seed(4001)
rand_pred_grid <- purrr::map2_dfr(beta_grid$beta_0,
                                  beta_grid$beta_1,
                                  calc_lin_predictor,
                                  xn = seq(-2, 2, by = 0.5)) %>% 
  mutate(y = rnorm(n = n(),
                   mean = mu,
                   sd = sigma_true))
```

The `mutate()` function performs several steps together in order to generate the desired number of random samples. First, the `n()` is assigned to the first argument of the `rnorm()` function. The `n()` function returns the total number of rows in the current data object. The `rnorm()` function generates random samples from a gaussian distribution with mean defined by the `mean` argument the standard deviation specified by the `sd` argument. The `mu` variable, which was created by the `calc_lin_predictor()` function is assigned to the `mean` argument. The standard deviation was assumed to be constant, set equal to `sigma_true =` 0.5.

The previous code chunk can be related to the mathmetical format of our model. The `calc_lin_predictor()` function determines the value of the linear predictor, ![\\mu](https://latex.codecogs.com/png.latex?%5Cmu "\mu"), given ![\\boldsymbol{\\beta}](https://latex.codecogs.com/png.latex?%5Cboldsymbol%7B%5Cbeta%7D "\boldsymbol{\beta}") and the input, ![x](https://latex.codecogs.com/png.latex?x "x"). We then sample ![y](https://latex.codecogs.com/png.latex?y "y") given ![\\mu](https://latex.codecogs.com/png.latex?%5Cmu "\mu") and ![\\sigma](https://latex.codecogs.com/png.latex?%5Csigma "\sigma"). In the specific setup we used, each ![x](https://latex.codecogs.com/png.latex?x "x") value within `rand_pred_grid` has exactly 1 random sample associated with it, for each ![\\beta\_{0}](https://latex.codecogs.com/png.latex?%5Cbeta_%7B0%7D "\beta_{0}") and ![\\beta\_{1}](https://latex.codecogs.com/png.latex?%5Cbeta_%7B1%7D "\beta_{1}") pair. To confirm that is the case, pipe `rand_pred_grid` into the `count()` function and calculate the unique pairs of ![\\boldsymbol{\\beta}](https://latex.codecogs.com/png.latex?%5Cboldsymbol%7B%5Cbeta%7D "\boldsymbol{\beta}") parameters, and then the unique ![x](https://latex.codecogs.com/png.latex?x "x") values. Since we used 9 specific ![x](https://latex.codecogs.com/png.latex?x "x") values and 9 unique combinations of the ![\\boldsymbol{\\beta}](https://latex.codecogs.com/png.latex?%5Cboldsymbol%7B%5Cbeta%7D "\boldsymbol{\beta}") parameters, we should see exactly 9 rows associated with each unique ![\\boldsymbol{\\beta}](https://latex.codecogs.com/png.latex?%5Cboldsymbol%7B%5Cbeta%7D "\boldsymbol{\beta}") and ![x](https://latex.codecogs.com/png.latex?x "x") value, respectively.

``` r
rand_pred_grid %>% 
  count(beta_0, beta_1)
```

    ## # A tibble: 9 x 3
    ##   beta_0 beta_1     n
    ##    <int>  <int> <int>
    ## 1     -1     -1     9
    ## 2     -1      0     9
    ## 3     -1      1     9
    ## 4      0     -1     9
    ## 5      0      0     9
    ## 6      0      1     9
    ## 7      1     -1     9
    ## 8      1      0     9
    ## 9      1      1     9

``` r
rand_pred_grid %>% 
  count(x)
```

    ## # A tibble: 9 x 2
    ##       x     n
    ##   <dbl> <int>
    ## 1  -2       9
    ## 2  -1.5     9
    ## 3  -1       9
    ## 4  -0.5     9
    ## 5   0       9
    ## 6   0.5     9
    ## 7   1       9
    ## 8   1.5     9
    ## 9   2       9

By dispalying our random draws on top of our "ribbon style" plot we can see our noise impacts the apparent relationship between the output and the input. Remember, in our example we are dealing with linear relationships. Any non-linear behavior is artificial. To start out, we continue to focus our attention on the ![\\beta\_{0}=1](https://latex.codecogs.com/png.latex?%5Cbeta_%7B0%7D%3D1 "\beta_{0}=1") and ![\\beta\_{1}=1](https://latex.codecogs.com/png.latex?%5Cbeta_%7B1%7D%3D1 "\beta_{1}=1") case. The red dots in the figure produced below correspond to our random draws. **Would you say the red dots come from a linear relationship?**

``` r
lin_pred_grid %>% 
  mutate(beta_0_plot = forcats::fct_rev(as.factor(beta_0))) %>% 
  filter(beta_0 == 1, beta_1 == 1) %>% 
  ggplot(mapping = aes(x = x, y = mu)) +
  geom_hline(yintercept = 0, color = "grey50", linetype = "dashed") +
  geom_vline(xintercept = 0, color = "grey50", linetype = "dashed") +
  geom_ribbon(mapping = aes(ymin = mu - 3*sigma_true,
                            ymax = mu + 3*sigma_true,
                            group = interaction(beta_0, beta_1)),
              fill = "dodgerblue", alpha = 0.25) +
  geom_ribbon(mapping = aes(ymin = mu - 2*sigma_true,
                            ymax = mu + 2*sigma_true,
                            group = interaction(beta_0, beta_1)),
              fill = "dodgerblue", alpha = 0.25) +
  geom_ribbon(mapping = aes(ymin = mu - 1*sigma_true,
                            ymax = mu + 1*sigma_true,
                            group = interaction(beta_0, beta_1)),
              fill = "dodgerblue", alpha = 0.25) +
  geom_line(mapping = aes(group = interaction(beta_0, beta_1)),
            size = 1.15) +
  geom_point(data = rand_pred_grid %>% 
               mutate(beta_0_plot = forcats::fct_rev(as.factor(beta_0))) %>% 
               filter(beta_0 == 1, beta_1 == 1),
             mapping = aes(x = x, y = y),
             color = "red", size = 2) +
  facet_grid(beta_0_plot ~ beta_1,
             labeller = label_bquote(rows = beta[0]*"="*.(as.numeric(as.character(beta_0_plot))),
                                     cols = beta[1]*"="*.(beta_1))) +
  labs(y = "y") +
  theme_bw() +
  theme(strip.text = element_text(size = 12))
```

![](lecture_07_github_files/figure-markdown_github/compare_ribbon_with_random-1.png)

The random draws associated with the other 8 ![\\boldsymbol{\\beta}](https://latex.codecogs.com/png.latex?%5Cboldsymbol%7B%5Cbeta%7D "\boldsymbol{\beta}") pairs are visualized in the figure below. Ask yourself again, would you expect those random draws are associated with a linear trend?

``` r
lin_pred_grid %>% 
  mutate(beta_0_plot = forcats::fct_rev(as.factor(beta_0))) %>% 
  ggplot(mapping = aes(x = x, y = mu)) +
  geom_hline(yintercept = 0, color = "grey50", linetype = "dashed") +
  geom_vline(xintercept = 0, color = "grey50", linetype = "dashed") +
  geom_ribbon(mapping = aes(ymin = mu - 3*sigma_true,
                            ymax = mu + 3*sigma_true,
                            group = interaction(beta_0, beta_1)),
              fill = "dodgerblue", alpha = 0.25) +
  geom_ribbon(mapping = aes(ymin = mu - 2*sigma_true,
                            ymax = mu + 2*sigma_true,
                            group = interaction(beta_0, beta_1)),
              fill = "dodgerblue", alpha = 0.25) +
  geom_ribbon(mapping = aes(ymin = mu - 1*sigma_true,
                            ymax = mu + 1*sigma_true,
                            group = interaction(beta_0, beta_1)),
              fill = "dodgerblue", alpha = 0.25) +
  geom_line(mapping = aes(group = interaction(beta_0, beta_1)),
            size = 1.15) +
  geom_point(data = rand_pred_grid %>% 
               mutate(beta_0_plot = forcats::fct_rev(as.factor(beta_0))),
             mapping = aes(x = x, y = y),
             color = "red", size = 2) +
  facet_grid(beta_0_plot ~ beta_1,
             labeller = label_bquote(rows = beta[0]*"="*.(as.numeric(as.character(beta_0_plot))),
                                     cols = beta[1]*"="*.(beta_1))) +
  labs(y = "y") +
  theme_bw() +
  theme(strip.text = element_text(size = 12))
```

![](lecture_07_github_files/figure-markdown_github/compare_ribbon_with_random_bb-1.png)

### Increasing sample size

If you do not think the red dots in the previous figures do not appear to be linear, you should then consider what happens as increase the sample size. So let's do that now. The code chunk below defines the ![\\boldsymbol{\\beta}](https://latex.codecogs.com/png.latex?%5Cboldsymbol%7B%5Cbeta%7D "\boldsymbol{\beta}") and input grid using the `expand.grid()` function. The fourth variable, `rep_id`, is used as an identifier for each unique *replicate*. So based on the syntax below, we will have each `beta_0`, `beta_1`, and `x` combination appearing 7 times. **Do you understand what actions take place in the `mutate()` call?**

``` r
set.seed(4002)
rand_grid_7 <- expand.grid(beta_0 = -1:1,
                           beta_1 = -1:1,
                           x = seq(-2, 2, by = 0.5),
                           rep_id = 1:7,
                           stringsAsFactors = FALSE,
                           KEEP.OUT.ATTRS = FALSE) %>% 
  as.data.frame() %>% tbl_df() %>% 
  mutate(mu = beta_0 + beta_1 * x,
         y = rnorm(n = n(), mu, sigma_true))
```

Display all 7 *replicate* samples at each of the ![x](https://latex.codecogs.com/png.latex?x "x") positions for one ![\\boldsymbol{\\beta}](https://latex.codecogs.com/png.latex?%5Cboldsymbol%7B%5Cbeta%7D "\boldsymbol{\beta}") pair. What do you think the mean of those 7 samples should be at each ![x](https://latex.codecogs.com/png.latex?x "x") position?

``` r
lin_pred_grid %>% 
  mutate(beta_0_plot = forcats::fct_rev(as.factor(beta_0))) %>% 
  filter(beta_0 == 1, beta_1 == 1) %>% 
  ggplot(mapping = aes(x = x, y = mu)) +
  geom_hline(yintercept = 0, color = "grey50", linetype = "dashed") +
  geom_vline(xintercept = 0, color = "grey50", linetype = "dashed") +
  geom_ribbon(mapping = aes(ymin = mu - 3*sigma_true,
                            ymax = mu + 3*sigma_true,
                            group = interaction(beta_0, beta_1)),
              fill = "dodgerblue", alpha = 0.25) +
  geom_ribbon(mapping = aes(ymin = mu - 2*sigma_true,
                            ymax = mu + 2*sigma_true,
                            group = interaction(beta_0, beta_1)),
              fill = "dodgerblue", alpha = 0.25) +
  geom_ribbon(mapping = aes(ymin = mu - 1*sigma_true,
                            ymax = mu + 1*sigma_true,
                            group = interaction(beta_0, beta_1)),
              fill = "dodgerblue", alpha = 0.25) +
  geom_line(mapping = aes(group = interaction(beta_0, beta_1)),
            size = 1.15) +
  geom_point(data = rand_grid_7 %>% 
               mutate(beta_0_plot = forcats::fct_rev(as.factor(beta_0))) %>% 
               filter(beta_0 == 1, beta_1 == 1),
             mapping = aes(x = x, y = y),
             color = "red", size = 2) +
  facet_grid(beta_0_plot ~ beta_1,
             labeller = label_bquote(rows = beta[0]*"="*.(as.numeric(as.character(beta_0_plot))),
                                     cols = beta[1]*"="*.(beta_1))) +
  labs(y = "y") +
  theme_bw() +
  theme(strip.text = element_text(size = 12))
```

![](lecture_07_github_files/figure-markdown_github/viz_7_reps_1_beta-1.png)

Calculate the empirical (sample) average at each ![x](https://latex.codecogs.com/png.latex?x "x") position and denote with a yellow triangle. The triangle points upward if the sample average is greater than ![\\mu](https://latex.codecogs.com/png.latex?%5Cmu "\mu"), while the triangle points down if the sample average is less than ![\\mu](https://latex.codecogs.com/png.latex?%5Cmu "\mu"). **Why is the sample average being compared with ![\\mu](https://latex.codecogs.com/png.latex?%5Cmu "\mu")?**

``` r
lin_pred_grid %>% 
  mutate(beta_0_plot = forcats::fct_rev(as.factor(beta_0))) %>% 
  filter(beta_0 == 1, beta_1 == 1) %>% 
  ggplot(mapping = aes(x = x, y = mu)) +
  geom_hline(yintercept = 0, color = "grey50", linetype = "dashed") +
  geom_vline(xintercept = 0, color = "grey50", linetype = "dashed") +
  geom_ribbon(mapping = aes(ymin = mu - 3*sigma_true,
                            ymax = mu + 3*sigma_true,
                            group = interaction(beta_0, beta_1)),
              fill = "dodgerblue", alpha = 0.25) +
  geom_ribbon(mapping = aes(ymin = mu - 2*sigma_true,
                            ymax = mu + 2*sigma_true,
                            group = interaction(beta_0, beta_1)),
              fill = "dodgerblue", alpha = 0.25) +
  geom_ribbon(mapping = aes(ymin = mu - 1*sigma_true,
                            ymax = mu + 1*sigma_true,
                            group = interaction(beta_0, beta_1)),
              fill = "dodgerblue", alpha = 0.25) +
  geom_line(mapping = aes(group = interaction(beta_0, beta_1)),
            size = 1.15) +
  geom_point(data = rand_grid_7 %>% 
               mutate(beta_0_plot = forcats::fct_rev(as.factor(beta_0))) %>% 
               filter(beta_0 == 1, beta_1 == 1),
             mapping = aes(x = x, y = y),
             color = "red", size = 2) +
  geom_point(data = rand_grid_7 %>% 
               group_by(beta_0, beta_1, x) %>% 
               summarise(avg_y = mean(y),
                         mu_val = first(mu)) %>% 
               ungroup() %>% 
               mutate(beta_0_plot = forcats::fct_rev(as.factor(beta_0))) %>% 
               filter(beta_0 == 1, beta_1 == 1),
               mapping = aes(x = x, y = avg_y,
                             shape = avg_y >= mu_val),
             fill = "gold", size = 2.5, color = "black") +
  facet_grid(beta_0_plot ~ beta_1,
             labeller = label_bquote(rows = beta[0]*"="*.(as.numeric(as.character(beta_0_plot))),
                                     cols = beta[1]*"="*.(beta_1))) +
  scale_shape_manual(guide = FALSE,
                     values = c("TRUE" = 24,
                                "FALSE" = 25)) +
  labs(y = "y") +
  theme_bw() +
  theme(strip.text = element_text(size = 12))
```

![](lecture_07_github_files/figure-markdown_github/viz_7_reps_1_beta_bb-1.png)

As we increase the sample size, what do you anticipate the sample averages will converge to? Let's check by simulating 25 random samples at each ![x](https://latex.codecogs.com/png.latex?x "x") position.

``` r
set.seed(4003)
rand_grid_25 <- expand.grid(beta_0 = -1:1,
                            beta_1 = -1:1,
                            x = seq(-2, 2, by = 0.5),
                            rep_id = 1:25,
                            stringsAsFactors = FALSE,
                            KEEP.OUT.ATTRS = FALSE) %>% 
  as.data.frame() %>% tbl_df() %>% 
  mutate(mu = beta_0 + beta_1 * x,
         y = rnorm(n = n(), mu, sigma_true))
```

Plot all of the random samples and the new sample averages:

``` r
lin_pred_grid %>% 
  mutate(beta_0_plot = forcats::fct_rev(as.factor(beta_0))) %>% 
  filter(beta_0 == 1, beta_1 == 1) %>% 
  ggplot(mapping = aes(x = x, y = mu)) +
  geom_hline(yintercept = 0, color = "grey50", linetype = "dashed") +
  geom_vline(xintercept = 0, color = "grey50", linetype = "dashed") +
  geom_ribbon(mapping = aes(ymin = mu - 3*sigma_true,
                            ymax = mu + 3*sigma_true,
                            group = interaction(beta_0, beta_1)),
              fill = "dodgerblue", alpha = 0.25) +
  geom_ribbon(mapping = aes(ymin = mu - 2*sigma_true,
                            ymax = mu + 2*sigma_true,
                            group = interaction(beta_0, beta_1)),
              fill = "dodgerblue", alpha = 0.25) +
  geom_ribbon(mapping = aes(ymin = mu - 1*sigma_true,
                            ymax = mu + 1*sigma_true,
                            group = interaction(beta_0, beta_1)),
              fill = "dodgerblue", alpha = 0.25) +
  geom_line(mapping = aes(group = interaction(beta_0, beta_1)),
            size = 1.15) +
  geom_point(data = rand_grid_25 %>% 
               mutate(beta_0_plot = forcats::fct_rev(as.factor(beta_0))) %>% 
               filter(beta_0 == 1, beta_1 == 1),
             mapping = aes(x = x, y = y),
             color = "red", size = 2) +
  geom_point(data = rand_grid_25 %>% 
               group_by(beta_0, beta_1, x) %>% 
               summarise(avg_y = mean(y),
                         mu_val = first(mu)) %>% 
               ungroup() %>% 
               mutate(beta_0_plot = forcats::fct_rev(as.factor(beta_0))) %>% 
               filter(beta_0 == 1, beta_1 == 1),
               mapping = aes(x = x, y = avg_y,
                             shape = avg_y >= mu_val),
             fill = "gold", size = 2.5, color = "black") +
  facet_grid(beta_0_plot ~ beta_1,
             labeller = label_bquote(rows = beta[0]*"="*.(as.numeric(as.character(beta_0_plot))),
                                     cols = beta[1]*"="*.(beta_1))) +
  scale_shape_manual(guide = FALSE,
                     values = c("TRUE" = 24,
                                "FALSE" = 25)) +
  labs(y = "y") +
  theme_bw() +
  theme(strip.text = element_text(size = 12))
```

![](lecture_07_github_files/figure-markdown_github/viz_25_reps_1_beta-1.png)

Let's now push the sample size to 1000 to see what what happens in the limit of many, many samples.

``` r
set.seed(4004)
rand_grid_1000 <- expand.grid(beta_0 = -1:1,
                              beta_1 = -1:1,
                              x = seq(-2, 2, by = 0.5),
                              rep_id = 1:1000,
                              stringsAsFactors = FALSE,
                              KEEP.OUT.ATTRS = FALSE) %>% 
  as.data.frame() %>% tbl_df() %>% 
  mutate(mu = beta_0 + beta_1 * x,
         y = rnorm(n = n(), mu, sigma_true))
```

Plot all 1000 random samples and the sample averages at each ![x](https://latex.codecogs.com/png.latex?x "x") position:

``` r
lin_pred_grid %>% 
  mutate(beta_0_plot = forcats::fct_rev(as.factor(beta_0))) %>% 
  filter(beta_0 == 1, beta_1 == 1) %>% 
  ggplot(mapping = aes(x = x, y = mu)) +
  geom_hline(yintercept = 0, color = "grey50", linetype = "dashed") +
  geom_vline(xintercept = 0, color = "grey50", linetype = "dashed") +
  geom_ribbon(mapping = aes(ymin = mu - 3*sigma_true,
                            ymax = mu + 3*sigma_true,
                            group = interaction(beta_0, beta_1)),
              fill = "dodgerblue", alpha = 0.25) +
  geom_ribbon(mapping = aes(ymin = mu - 2*sigma_true,
                            ymax = mu + 2*sigma_true,
                            group = interaction(beta_0, beta_1)),
              fill = "dodgerblue", alpha = 0.25) +
  geom_ribbon(mapping = aes(ymin = mu - 1*sigma_true,
                            ymax = mu + 1*sigma_true,
                            group = interaction(beta_0, beta_1)),
              fill = "dodgerblue", alpha = 0.25) +
  geom_line(mapping = aes(group = interaction(beta_0, beta_1)),
            size = 1.15) +
  geom_point(data = rand_grid_1000 %>% 
               mutate(beta_0_plot = forcats::fct_rev(as.factor(beta_0))) %>% 
               filter(beta_0 == 1, beta_1 == 1),
             mapping = aes(x = x, y = y),
             color = "red", size = 2) +
  geom_point(data = rand_grid_1000 %>% 
               group_by(beta_0, beta_1, x) %>% 
               summarise(avg_y = mean(y),
                         mu_val = first(mu)) %>% 
               ungroup() %>% 
               mutate(beta_0_plot = forcats::fct_rev(as.factor(beta_0))) %>% 
               filter(beta_0 == 1, beta_1 == 1),
               mapping = aes(x = x, y = avg_y,
                             shape = avg_y >= mu_val),
             fill = "gold", size = 2.5, color = "black") +
  facet_grid(beta_0_plot ~ beta_1,
             labeller = label_bquote(rows = beta[0]*"="*.(as.numeric(as.character(beta_0_plot))),
                                     cols = beta[1]*"="*.(beta_1))) +
  scale_shape_manual(guide = FALSE,
                     values = c("TRUE" = 24,
                                "FALSE" = 25)) +
  labs(y = "y") +
  theme_bw() +
  theme(strip.text = element_text(size = 12))
```

![](lecture_07_github_files/figure-markdown_github/viz_1000_reps_1_beta-1.png)

The sample averages are convering to the linear predictor! But, what else can we say about the samples? What shape do the distributions of the random samples take, at each ![x](https://latex.codecogs.com/png.latex?x "x") position? The figure below plots the histogram of the 1000 random ![y](https://latex.codecogs.com/png.latex?y "y") samples at each ![x](https://latex.codecogs.com/png.latex?x "x") position per subplot. All of the subplots correspond to the case of ![\\beta\_{0} = 1](https://latex.codecogs.com/png.latex?%5Cbeta_%7B0%7D%20%3D%201 "\beta_{0} = 1") and ![\\beta\_{1} = 1](https://latex.codecogs.com/png.latex?%5Cbeta_%7B1%7D%20%3D%201 "\beta_{1} = 1"). All of the distributions are essentially Gaussians, because the likelihood is Gaussian!

``` r
rand_grid_1000 %>% 
  filter(beta_0 == 1, beta_1 == 1) %>% 
  ggplot(mapping = aes(x = y)) +
  geom_histogram(bins = 35, 
                 mapping = aes(group = x)) +
  facet_wrap(~ x, labeller = "label_both") +
  theme_bw() +
  theme(axis.text.y = element_blank())
```

![](lecture_07_github_files/figure-markdown_github/viz_dist_shape_1000_reps-1.png)

Model fitting
-------------

Now that we are comfortable with the model, let's focus on "fitting" the model. We observe ![N](https://latex.codecogs.com/png.latex?N "N") pairs of the input and the response, ![\\{x\_n, y\_n\\}](https://latex.codecogs.com/png.latex?%5C%7Bx_n%2C%20y_n%5C%7D "\{x_n, y_n\}"). Based on those observations, what can we say about the ![\\boldsymbol{\\beta}](https://latex.codecogs.com/png.latex?%5Cboldsymbol%7B%5Cbeta%7D "\boldsymbol{\beta}") and ![\\sigma](https://latex.codecogs.com/png.latex?%5Csigma "\sigma") parameters? Well, let's start out by assuming that ![\\sigma](https://latex.codecogs.com/png.latex?%5Csigma "\sigma") is known. Thus, we have to learn only **two** unknown parameters, ![\\beta\_{0}](https://latex.codecogs.com/png.latex?%5Cbeta_%7B0%7D "\beta_{0}") and ![\\beta\_{1}](https://latex.codecogs.com/png.latex?%5Cbeta_%7B1%7D "\beta_{1}").

### Synthetic data

We will generate synthetic data from specified *true* parameter values. We will then *learn* or *update* the parameters based on the observations and compare our *updated beliefs* with the known true parameter values. This verification style, *method of manufacturing solutions* approach is very useful making sure we setup the learning problem correctly. We know all aspects of the *data generating process* which produces the observations, and so we can check if there are errors in our code, as well as study impacts of noise and sample size on our ability to learn the true values which created the data.

#### Problem specification

We will assume the true parameters of the data generating process are, ![\\beta\_{0, \\mathrm{true}} = -0.25](https://latex.codecogs.com/png.latex?%5Cbeta_%7B0%2C%20%5Cmathrm%7Btrue%7D%7D%20%3D%20-0.25 "\beta_{0, \mathrm{true}} = -0.25"), ![\\beta\_{1, \\mathrm{true}} = 1.15](https://latex.codecogs.com/png.latex?%5Cbeta_%7B1%2C%20%5Cmathrm%7Btrue%7D%7D%20%3D%201.15 "\beta_{1, \mathrm{true}} = 1.15"), and ![\\sigma\_{\\mathrm{true}} = 0.5](https://latex.codecogs.com/png.latex?%5Csigma_%7B%5Cmathrm%7Btrue%7D%7D%20%3D%200.5 "\sigma_{\mathrm{true}} = 0.5"). We will assume that the input variable, ![x](https://latex.codecogs.com/png.latex?x "x"), has a standard normal distribution. Thus, our ![N](https://latex.codecogs.com/png.latex?N "N") *training points* will not be uniformly spaced between some lower and upper bound on the input variable.

The code chunk below generates 100 random observations of the input, ![x](https://latex.codecogs.com/png.latex?x "x"), and then calculates the *true* linear predictor value based on the assumed *true* parameter values. Random observations are then generated using the assumed *true* ![\\sigma](https://latex.codecogs.com/png.latex?%5Csigma "\sigma") value.

``` r
### set the true linear predictor coefficients
beta_0_true <- -0.25
beta_1_true <- 1.15

### generate random input values
set.seed(4100)
x_demo <- rnorm(n = 100, mean = 0, sd = 1)

### evaluate the linear predictor and generate
### random, noisy observations
demo_df <- tibble::tibble(
  x = x_demo
) %>% 
  mutate(mu = beta_0_true + beta_1_true * x,
         y = rnorm(n = n(), mean = mu, sd = sigma_true))
```

All 100 observations are plotted as a scatter plot between the response and the input below. The observations are shown as red dots. For context, the *true* linear predictor and *true* uncertainty intervals (similar to the previous ribbon figures) are dispalyed behind the observations.

``` r
### create the fine grid for visualization
demo_fine <- tibble::tibble(
  x = seq(-2.5, 2.5, length.out = 101)
) %>% 
  mutate(mu = beta_0_true + beta_1_true * x)

### visualize the observations relative to the true data
### generating process
demo_fine %>% 
  ggplot(mapping = aes(x = x)) +
  geom_ribbon(mapping = aes(ymin = mu - 3 * sigma_true,
                            ymax = mu + 3 * sigma_true,
                            group = 1),
              fill = "dodgerblue", alpha = 0.25) +
  geom_ribbon(mapping = aes(ymin = mu - 2 * sigma_true,
                            ymax = mu + 2 * sigma_true,
                            group = 1),
              fill = "dodgerblue", alpha = 0.25) +
  geom_ribbon(mapping = aes(ymin = mu - 1 * sigma_true,
                            ymax = mu + 1 * sigma_true,
                            group = 1),
              fill = "dodgerblue", alpha = 0.25) +
  geom_line(mapping = aes(y = mu,
                          group = 1),
            color = "black", size = 1.15) +
  geom_point(data = demo_df,
             mapping = aes(y = y),
             color = "red", size = 2) +
  labs(x = "x", y = "y") +
  theme_bw() +
  theme(axis.title = element_text(size = 11))
```

![](lecture_07_github_files/figure-markdown_github/viz_toy_demo-1.png)

Rather than working with all 100 observations out of the gate, let's start with the first 10 observations. Since all of the input values were generated randomly, we can say in this scenario that we randomly selected the first 11 observations. the figure below repeats the previous figure, but now marks the first 10 points with red, and the remaining observations in dark grey.

``` r
demo_fine %>% 
  ggplot(mapping = aes(x = x)) +
  geom_ribbon(mapping = aes(ymin = mu - 3 * sigma_true,
                            ymax = mu + 3 * sigma_true,
                            group = 1),
              fill = "dodgerblue", alpha = 0.25) +
  geom_ribbon(mapping = aes(ymin = mu - 2 * sigma_true,
                            ymax = mu + 2 * sigma_true,
                            group = 1),
              fill = "dodgerblue", alpha = 0.25) +
  geom_ribbon(mapping = aes(ymin = mu - 1 * sigma_true,
                            ymax = mu + 1 * sigma_true,
                            group = 1),
              fill = "dodgerblue", alpha = 0.25) +
  geom_line(mapping = aes(y = mu,
                          group = 1),
            color = "black", size = 1.15) +
  geom_point(data = demo_df %>% 
               tibble::rowid_to_column("obs_id"),
             mapping = aes(y = y,
                           color = obs_id > 10),
             size = 2) +
  scale_color_manual("",
                     values = c("TRUE" = "grey30",
                                "FALSE" = "red"),
                     labels = c("TRUE" = "last 90 points",
                                "FALSE" = "first 10 points")) +
  labs(x = "x", y = "y") +
  theme_bw() +
  theme(axis.title = element_text(size = 11),
        legend.position = "top")
```

![](lecture_07_github_files/figure-markdown_github/viz_toy_demo_bb-1.png)

Let's subset the complete set of observations into just the first 10 points and name that as the training set.

``` r
train_df <- demo_df %>% 
  tibble::rowid_to_column("obs_id") %>% 
  slice(1:10)
```

### Learning

We will follow a complete Bayesian formulation for the learning problem. We will therefore update our prior belief about the unknown parameters given the observations. After updating, our posterior belief is represented by the posterior distribution. The posterior distribution can be written as being proportional to:

![ 
p\\left(\\boldsymbol{\\beta} \\mid \\mathrm{y}, \\mathrm{x}, \\sigma \\right) \\propto p\\left( \\mathrm{y} \\mid \\mathrm{x}, \\boldsymbol{\\beta}, \\sigma \\right) p\\left( \\boldsymbol{\\beta} \\right)
](https://latex.codecogs.com/png.latex?%20%0Ap%5Cleft%28%5Cboldsymbol%7B%5Cbeta%7D%20%5Cmid%20%5Cmathrm%7By%7D%2C%20%5Cmathrm%7Bx%7D%2C%20%5Csigma%20%5Cright%29%20%5Cpropto%20p%5Cleft%28%20%5Cmathrm%7By%7D%20%5Cmid%20%5Cmathrm%7Bx%7D%2C%20%5Cboldsymbol%7B%5Cbeta%7D%2C%20%5Csigma%20%5Cright%29%20p%5Cleft%28%20%5Cboldsymbol%7B%5Cbeta%7D%20%5Cright%29%0A " 
p\left(\boldsymbol{\beta} \mid \mathrm{y}, \mathrm{x}, \sigma \right) \propto p\left( \mathrm{y} \mid \mathrm{x}, \boldsymbol{\beta}, \sigma \right) p\left( \boldsymbol{\beta} \right)
")

#### Prior specification

We must specify our prior beliefs about the unknown ![\\boldsymbol{\\beta}](https://latex.codecogs.com/png.latex?%5Cboldsymbol%7B%5Cbeta%7D "\boldsymbol{\beta}") parameters. Let's keep things simple for now, and assume independent standard normals. The prior can therefore be written as:

![ 
p\\left(\\boldsymbol{\\beta}\\right) = p\\left(\\beta\_{0}\\right) p\\left( \\beta\_{1} \\right) = \\mathrm{normal}\\left( \\beta\_{0} \\mid 0, 1 \\right) \\mathrm{normal}\\left(\\beta\_{1} \\mid 0,1 \\right)
](https://latex.codecogs.com/png.latex?%20%0Ap%5Cleft%28%5Cboldsymbol%7B%5Cbeta%7D%5Cright%29%20%3D%20p%5Cleft%28%5Cbeta_%7B0%7D%5Cright%29%20p%5Cleft%28%20%5Cbeta_%7B1%7D%20%5Cright%29%20%3D%20%5Cmathrm%7Bnormal%7D%5Cleft%28%20%5Cbeta_%7B0%7D%20%5Cmid%200%2C%201%20%5Cright%29%20%5Cmathrm%7Bnormal%7D%5Cleft%28%5Cbeta_%7B1%7D%20%5Cmid%200%2C1%20%5Cright%29%0A " 
p\left(\boldsymbol{\beta}\right) = p\left(\beta_{0}\right) p\left( \beta_{1} \right) = \mathrm{normal}\left( \beta_{0} \mid 0, 1 \right) \mathrm{normal}\left(\beta_{1} \mid 0,1 \right)
")

How should we expect this prior to influence our learning? Well, for either parameter, we can calculate the prior probability the parameter is below/above certain threshold values. For example, the prior probability that the slope is less than or equal to a value of 2:

``` r
pnorm(2, mean = 0, sd = 1)
```

    ## [1] 0.9772499

Since we are using Gaussian distributions, our prior probability is symmetric about the prior mean. Thus, we are placing equal prior chance on positive slopes, as we are negative slopes. Depending on the context of the problem, we might want to favor one over the other. For now, we will continue to use our standard normal.

The questions and statements we just discussed were in terms of a single parameter. Our prior though encomposses both parameters within the model, the slope and the intercept. To consider how our assumption impacts the two of them together, let's visualize the *joint* distribution. **Since we have already specified we are assuming the parameters are independent *a priori* what do you expect the prior distribution looks like?**

Before visualizing the joint prior distribution, let's introduce the `"matrix"` data type. We used the `"matrix"` last week when we covered the Laplace approximation, but we did not really discuss it. In `R`, `data.frame`s and `tibble`s, and lists can be thought of as "containers" of data. Although we can perform math on variables within a `data.frame`, or even apply a mathematical operation to all of the columns of a `data.frame`, we **cannot** perform linear algebra operations with them. Linear algebra is reserved for `"vector"`s and `"matrix"` data types.

The code chunk below creates a simple ![2 \\times 2](https://latex.codecogs.com/png.latex?2%20%5Ctimes%202 "2 \times 2") matrix. The elements consist of the integers 1 through 4. This setup helps show you how a matrix is "filled", when it is created from a vector. Notice that I set the `byrow` argument to be `TRUE`. This means that the matrix fills along a row, rather than down a column.

``` r
matrix(c(1, 2, 3, 4), nrow = 2, byrow = TRUE)
```

    ##      [,1] [,2]
    ## [1,]    1    2
    ## [2,]    3    4

Let's now define the prior mean vector and prior covariance matrix for our assumed prior.

``` r
my_prior_mean <- c(0, 0)
my_prior_cov <- matrix(c(1, 0, 0, 1), nrow = 2, byrow = TRUE)
```

Let's define a grid of possible parameter values, and then evaluate the log-prior density over that grid.

``` r
### create the grid of intercept and slope values
beta_param_grid <- expand.grid(beta_0 = seq(-4, 4, length.out = 251),
                               beta_1 = seq(-4, 4, length.out = 251),
                               KEEP.OUT.ATTRS = FALSE,
                               stringsAsFactors = FALSE) %>% 
  as.data.frame() %>% tbl_df()

### evaluate the log-prior density with `mvtnorm::dmvnorm()`
beta_prior_log_density <- mvtnorm::dmvnorm(as.matrix(beta_param_grid),
                                           mean = my_prior_mean,
                                           sigma = my_prior_cov,
                                           log = TRUE)

### bring the result to the grid of values
beta_prior_grid <- beta_param_grid %>% 
  mutate(log_prior = beta_prior_log_density)
```

We can now visualize the log-prior density surface. The color scheme is consistent with what we used last week. Grey regions correspond to ![\\beta\_{0}](https://latex.codecogs.com/png.latex?%5Cbeta_%7B0%7D "\beta_{0}") and ![\\beta\_{1}](https://latex.codecogs.com/png.latex?%5Cbeta_%7B1%7D "\beta_{1}") values with less than 0.01% prior probability. **Is this what you expected the prior to look like?** Because this is a synthetic data problem, the *true* parameter values are shown as vertical and horizontal red dashed lines.

``` r
beta_prior_grid %>% 
  mutate(log_prior_2 = log_prior - max(log_prior)) %>% 
  ggplot(mapping = aes(x = beta_1, y = beta_0)) +
  geom_raster(mapping = aes(fill = log_prior_2)) +
  stat_contour(mapping = aes(z = log_prior_2),
               breaks = log(c(0.01/100, 0.01, 0.1, 0.5, 0.9)),
               size = 2.2,
               color = "black") +
  geom_hline(yintercept = beta_0_true,
             color = "red", linetype = "dashed") +
  geom_vline(xintercept = beta_1_true,
             color = "red", linetype = "dashed") +
  coord_fixed(ratio = 1) +
  scale_fill_viridis_c(guide = FALSE, option = "viridis",
                       limits = log(c(0.01/100, 1.0))) +
  labs(x = expression(beta[1]), y = expression(beta[0])) +
  theme_bw() +
  theme(legend.position = "top",
        axis.text = element_text(size = 12),
        axis.title = element_text(size = 12))
```

![](lecture_07_github_files/figure-markdown_github/viz_log_prior-1.png)

#### Probability model

We are now in a position to write out the complete probability model accounting for the likelihood and the prior. We will write the complete model in terms of the individual likelihoods associated with each observation. The format used in the equation block below is that the top line is the ![n](https://latex.codecogs.com/png.latex?n "n")-th likelihood. The second line down gives the deterministic relationship. After that, the priors on all parameters are specified. The model is:

![ 
y\_{n} \\mid \\mu\_{n}, \\sigma \\sim \\mathrm{normal}\\left(y\_{n} \\mid \\mu\_{n}, \\sigma \\right) \\\\ \\mu\_{n} = \\beta\_{0} + \\beta\_{1}x\_{n} \\\\ \\beta\_{0} \\sim \\mathrm{normal}\\left( \\beta\_{0} \\mid 0, 1 \\right) \\\\ \\beta\_{1} \\sim \\mathrm{normal}\\left( \\beta\_{1} \\mid 0, 1 \\right)
](https://latex.codecogs.com/png.latex?%20%0Ay_%7Bn%7D%20%5Cmid%20%5Cmu_%7Bn%7D%2C%20%5Csigma%20%5Csim%20%5Cmathrm%7Bnormal%7D%5Cleft%28y_%7Bn%7D%20%5Cmid%20%5Cmu_%7Bn%7D%2C%20%5Csigma%20%5Cright%29%20%5C%5C%20%5Cmu_%7Bn%7D%20%3D%20%5Cbeta_%7B0%7D%20%2B%20%5Cbeta_%7B1%7Dx_%7Bn%7D%20%5C%5C%20%5Cbeta_%7B0%7D%20%5Csim%20%5Cmathrm%7Bnormal%7D%5Cleft%28%20%5Cbeta_%7B0%7D%20%5Cmid%200%2C%201%20%5Cright%29%20%5C%5C%20%5Cbeta_%7B1%7D%20%5Csim%20%5Cmathrm%7Bnormal%7D%5Cleft%28%20%5Cbeta_%7B1%7D%20%5Cmid%200%2C%201%20%5Cright%29%0A " 
y_{n} \mid \mu_{n}, \sigma \sim \mathrm{normal}\left(y_{n} \mid \mu_{n}, \sigma \right) \\ \mu_{n} = \beta_{0} + \beta_{1}x_{n} \\ \beta_{0} \sim \mathrm{normal}\left( \beta_{0} \mid 0, 1 \right) \\ \beta_{1} \sim \mathrm{normal}\left( \beta_{1} \mid 0, 1 \right)
")

#### Log-posterior

How can we "fit" the parameters in our model, given the observed data? Last week we saw two potential fitting schemes. The first was the grid approximation, which evaluates the log-posterior over a fine grid of possible parameter values. The second was the Laplace approximation, which approximated the arbitrary posterior with a multivariate normal (MVN) distribution. We saw both techniques in the context of the fitting distributions, but both methods are general. They can both be applied to fitting models as well!

As we also discussed last week, the grid approximation does not scale well to more than 2, maybe 3 parameters. Since we only have two parameters at the moment, we can apply the grid approximation to our problem. After visualizing the "true" log-posterior surface with the grid approximation, we will apply the Laplace approximation to this problem.

In order to apply either technique, we need to create a function which evaluates the log-posterior. The code chunk below, does just that, using the format and nomenclature from last week. The function `lm_logpost_01()` has two input arguments: `theta` and `my_info`. The ![\\boldsymbol{\\beta}](https://latex.codecogs.com/png.latex?%5Cboldsymbol%7B%5Cbeta%7D "\boldsymbol{\beta}") parameter values are contained within the `theta` input vector argument, while the list `my_info` contains all other information required to evaluate the log-posterior.

``` r
lm_logpost_01 <- function(theta, my_info)
{
  # unpack the parameter vector
  beta_0 <- theta[1]
  beta_1 <- theta[2]
  
  # calculate linear predictor
  mu <- beta_0 + beta_1 * my_info$xobs
  
  # evaluate the log-likelihood
  log_lik <- sum(dnorm(x = my_info$yobs,
                       mean = mu,
                       sd = my_info$sigma,
                       log = TRUE))
  
  # evaluate the log-prior
  log_prior <- dnorm(x = beta_0,
                     mean = my_info$b0_mu,
                     sd = my_info$b0_sd,
                     log = TRUE) +
    dnorm(x = beta_1,
          mean = my_info$b1_mu,
          sd = my_info$b1_sd,
          log = TRUE)
  
  # sum together
  log_lik + log_prior
}
```

Next, create a wrapper function to manage the execution of the log-posterior over a grid of parameter values.

``` r
eval_logpost_01 <- function(b0_val, b1_val, my_info)
{
  lm_logpost_01(c(b0_val, b1_val), my_info)
}
```

Before executing the function, we need to define our list of information. Let's use the data set consisting of the first 10 points.

``` r
info_use_01 <- list(
  xobs = train_df$x,
  yobs = train_df$y,
  sigma = sigma_true,
  b0_mu = 0,
  b0_sd = 1,
  b1_mu = 0,
  b1_sd = 1
)
```

Let's now evaluate the log-posterior over our prior grid of ![\\boldsymbol{\\beta}](https://latex.codecogs.com/png.latex?%5Cboldsymbol%7B%5Cbeta%7D "\boldsymbol{\beta}") values defined in the `beta_param_grid` object. Loop over all of the combinations of the ![\\beta\_{0}](https://latex.codecogs.com/png.latex?%5Cbeta_%7B0%7D "\beta_{0}") and ![\\beta\_{1}](https://latex.codecogs.com/png.latex?%5Cbeta_%7B1%7D "\beta_{1}") parameters with the `purrr` package, as shown below.

``` r
log_post_result_01 <- purrr::map2_dbl(beta_param_grid$beta_0,
                                      beta_param_grid$beta_1,
                                      eval_logpost_01,
                                      my_info = info_use_01)
```

We can now plot the log-posterior surface in the style that we visualized the log-prior surface. As shown in the figure below, the posterior is highly concentrated around the *true* ![\\boldsymbol{\\beta}](https://latex.codecogs.com/png.latex?%5Cboldsymbol%7B%5Cbeta%7D "\boldsymbol{\beta}") values after just 10 observations.

``` r
beta_prior_grid %>% 
  mutate(log_post = log_post_result_01) %>% 
  mutate(log_post_2 = log_post - max(log_post)) %>% 
  ggplot(mapping = aes(x = beta_1, y = beta_0)) +
  geom_raster(mapping = aes(fill = log_post_2)) +
  stat_contour(mapping = aes(z = log_post_2),
               breaks = log(c(0.01/100, 0.01, 0.1, 0.5, 0.9)),
               size = 1.05,
               color = "black") +
  geom_hline(yintercept = beta_0_true,
             color = "red", linetype = "dashed") +
  geom_vline(xintercept = beta_1_true,
             color = "red", linetype = "dashed") +
  coord_fixed(ratio = 1) +
  scale_fill_viridis_c(guide = FALSE, option = "viridis",
                       limits = log(c(0.01/100, 1.0))) +
  labs(x = expression(beta[1]), y = expression(beta[0]),
       title = "Log-posterior based on N = 10 observations") +
  theme_bw() +
  theme(legend.position = "top",
        axis.text = element_text(size = 12),
        axis.title = element_text(size = 12))
```

![](lecture_07_github_files/figure-markdown_github/viz_log_post-1.png)

We will now sequentially add one data point at a time, thereby allowing us to "watch" as we moved from the prior to the posterior. This way we can visualize the impact of each observation on the posterior. The figure below repeats the scatter plot between the input and the response, but this time prints the observation index number associated with each of the 10 training points.

``` r
demo_fine %>% 
  ggplot(mapping = aes(x = x)) +
  geom_ribbon(mapping = aes(ymin = mu - 3 * sigma_true,
                            ymax = mu + 3 * sigma_true,
                            group = 1),
              fill = "dodgerblue", alpha = 0.25) +
  geom_ribbon(mapping = aes(ymin = mu - 2 * sigma_true,
                            ymax = mu + 2 * sigma_true,
                            group = 1),
              fill = "dodgerblue", alpha = 0.25) +
  geom_ribbon(mapping = aes(ymin = mu - 1 * sigma_true,
                            ymax = mu + 1 * sigma_true,
                            group = 1),
              fill = "dodgerblue", alpha = 0.25) +
  geom_line(mapping = aes(y = mu,
                          group = 1),
            color = "black", size = 1.15) +
  geom_point(data = demo_df %>% 
               tibble::rowid_to_column("obs_id"),
             mapping = aes(y = y,
                           color = obs_id > 10,
                           size = obs_id > 10)) +
  geom_text(data = demo_df %>% 
              tibble::rowid_to_column("obs_id") %>% 
              filter(obs_id < 11),
            mapping = aes(y = y,
                          label = obs_id),
            color = "white") +
  scale_color_manual("",
                     values = c("TRUE" = "grey30",
                                "FALSE" = "red"),
                     labels = c("TRUE" = "last 90 points",
                                "FALSE" = "first 10 points")) +
  scale_size_manual("",
                    values = c("FALSE" = 5.5,
                               "TRUE" = 1.05),
                    labels = c("TRUE" = "last 90 points",
                                "FALSE" = "first 10 points")) +
  labs(x = "x", y = "y") +
  theme_bw() +
  theme(axis.title = element_text(size = 11),
        legend.position = "top")
```

![](lecture_07_github_files/figure-markdown_github/viz_toy_demo_data_cc-1.png)

Define a new wrapper function which manages the book keeping and sequential evaluationg of the log-posterior:

``` r
manage_log_post_01 <- function(num_obs, avail_data, my_settings, grid_use)
{
  # pass in the correct number of input/output observations
  my_settings$xobs <- avail_data$x[1:num_obs]
  my_settings$yobs <- avail_data$y[1:num_obs]
  
  lp_val <- purrr::map2_dbl(grid_use$beta_0,
                            grid_use$beta_1,
                            eval_logpost_01,
                            my_info = my_settings)
  
  grid_use %>% 
    mutate(log_post = lp_val,
           N = num_obs)
}
```

After defining a list containing the hyperparameters for the problem, the code chunk below executes the grid approximation by sequentially adding the first 10 data points.

``` r
hyper_list_01 <- list(
  sigma = sigma_true,
  b0_mu = 0,
  b0_sd = 1,
  b1_mu = 0,
  b1_sd = 1
)

log_post_surface_result_01 <- purrr::map_dfr(1:10,
                                             manage_log_post_01,
                                             avail_data = demo_df,
                                             my_settings = hyper_list_01,
                                             grid_use = beta_param_grid)
```

We can now visualize the contour plot of the log-posterior based on the number of observations. Include the prior as the ![N=0](https://latex.codecogs.com/png.latex?N%3D0 "N=0") case. As shown below, it appears that by ![N=5](https://latex.codecogs.com/png.latex?N%3D5 "N=5"), the posterior distribution is centered around the *true* parameter values.

``` r
beta_prior_grid %>% 
  rename(log_post = log_prior) %>% 
  mutate(N = 0) %>% 
  bind_rows(log_post_surface_result_01) %>% 
  group_by(N) %>% 
  mutate(max_log_post = max(log_post)) %>% 
  ungroup() %>% 
  mutate(log_post_2 = log_post - max_log_post) %>% 
  ggplot(mapping = aes(x = beta_1, y = beta_0)) +
  stat_contour(mapping = aes(z = log_post_2,
                             group = N),
               breaks = log(c(0.01/100, 0.01, 0.1, 0.5, 0.9)),
               size = 1.05,
               color = "black") +
  geom_hline(yintercept = beta_0_true,
             color = "red", linetype = "dashed") +
  geom_vline(xintercept = beta_1_true,
             color = "red", linetype = "dashed") +
  coord_fixed(ratio = 1) +
  facet_wrap(~N, labeller = "label_both") +
  scale_fill_viridis_c(guide = FALSE, option = "viridis",
                       limits = log(c(0.01/100, 1.0))) +
  labs(x = expression(beta[1]), y = expression(beta[0])) +
  theme_bw() +
  theme(legend.position = "top",
        axis.text = element_text(size = 12),
        axis.title = element_text(size = 12))
```

![](lecture_07_github_files/figure-markdown_github/viz_log_post_based_on_N-1.png)

To understand the posterior behavior a little more, let's focus on the ![N=3](https://latex.codecogs.com/png.latex?N%3D3 "N=3") case. The log-posterior surface associated with ![N=3](https://latex.codecogs.com/png.latex?N%3D3 "N=3") is shown below. For additional context, horizontal and vertical light blue lines correspond to the prior mean values.

``` r
beta_prior_grid %>% 
  rename(log_post = log_prior) %>% 
  mutate(N = 0) %>% 
  bind_rows(log_post_surface_result_01) %>% 
  group_by(N) %>% 
  mutate(max_log_post = max(log_post)) %>% 
  ungroup() %>% 
  mutate(log_post_2 = log_post - max_log_post) %>% 
  filter(N == 3) %>% 
  ggplot(mapping = aes(x = beta_1, y = beta_0)) +
  stat_contour(mapping = aes(z = log_post_2,
                             group = N),
               breaks = log(c(0.01/100, 0.01, 0.1, 0.5, 0.9)),
               size = 1.05,
               color = "black") +
  geom_hline(yintercept = info_use_01$b0_mu,
             color = "dodgerblue") +
  geom_vline(xintercept = info_use_01$b1_mu,
             color = "dodgerblue") +
  geom_hline(yintercept = beta_0_true,
             color = "red", linetype = "dashed") +
  geom_vline(xintercept = beta_1_true,
             color = "red", linetype = "dashed") +
  coord_fixed(ratio = 1) +
  facet_wrap(~N, labeller = "label_both") +
  scale_fill_viridis_c(guide = FALSE, option = "viridis",
                       limits = log(c(0.01/100, 1.0))) +
  labs(x = expression(beta[1]), y = expression(beta[0])) +
  theme_bw() +
  theme(legend.position = "top",
        axis.text = element_text(size = 12),
        axis.title = element_text(size = 12))
```

![](lecture_07_github_files/figure-markdown_github/viz_log_post_based_on_N_3-1.png)

Specify a subset of the complete grid, and plot those identified points on the log-posterior surface for the ![N=3](https://latex.codecogs.com/png.latex?N%3D3 "N=3") case.

``` r
### identify the distinct/unique values of beta_0 and beta_1
### within the zoomed in region for N = 3
focus_beta_grid <- beta_param_grid %>% 
  filter(between(beta_0, -2, 1.5) &
           between(beta_1, -0.5, 2.5))

focus_unique_b0 <- focus_beta_grid %>% 
  distinct(beta_0) %>% 
  arrange(beta_0)

focus_unique_b1 <- focus_beta_grid %>% 
  distinct(beta_1) %>% 
  arrange(beta_1)

### take every 10-th point in each
focus_unique_b0_subset <- focus_unique_b0 %>% 
  slice(seq(1, nrow(focus_unique_b0), by = 10))

focus_unique_b1_subset <- focus_unique_b1 %>% 
  slice(seq(1, nrow(focus_unique_b1), by = 10))

### identify the points to use
isolate_subset_grid_01 <- log_post_surface_result_01 %>% 
  group_by(N) %>% 
  mutate(max_log_post = max(log_post)) %>% 
  ungroup() %>% 
  mutate(log_post_2 = log_post - max_log_post) %>% 
  filter(N == 3) %>% 
  filter(beta_0 %in% focus_unique_b0_subset$beta_0 &
           beta_1 %in% focus_unique_b1_subset$beta_1)
```

``` r
beta_prior_grid %>% 
  rename(log_post = log_prior) %>% 
  mutate(N = 0) %>% 
  bind_rows(log_post_surface_result_01) %>% 
  group_by(N) %>% 
  mutate(max_log_post = max(log_post)) %>% 
  ungroup() %>% 
  mutate(log_post_2 = log_post - max_log_post) %>% 
  filter(N == 3) %>% 
  ggplot(mapping = aes(x = beta_1, y = beta_0)) +
  stat_contour(mapping = aes(z = log_post_2,
                             group = N),
               breaks = log(c(0.01/100, 0.01, 0.1, 0.5, 0.9)),
               size = 1.05,
               color = "black") +
  geom_point(data = isolate_subset_grid_01,
             color = "navyblue", size = 4.25) +
  geom_hline(yintercept = info_use_01$b0_mu,
             color = "dodgerblue") +
  geom_vline(xintercept = info_use_01$b1_mu,
             color = "dodgerblue") +
  geom_hline(yintercept = beta_0_true,
             color = "red", linetype = "dashed") +
  geom_vline(xintercept = beta_1_true,
             color = "red", linetype = "dashed") +
  coord_fixed(ratio = 1) +
  facet_wrap(~N, labeller = "label_both") +
  scale_fill_viridis_c(guide = FALSE, option = "viridis",
                       limits = log(c(0.01/100, 1.0))) +
  labs(x = expression(beta[1]), y = expression(beta[0])) +
  theme_bw() +
  theme(legend.position = "top",
        axis.text = element_text(size = 12),
        axis.title = element_text(size = 12))
```

![](lecture_07_github_files/figure-markdown_github/viz_logpost_surface_subset_01-1.png)

Color each of the markers based on the log-posterior at each ![\\boldsymbol{\\beta}](https://latex.codecogs.com/png.latex?%5Cboldsymbol%7B%5Cbeta%7D "\boldsymbol{\beta}") pair. The plot below is essentially a discretized version of the log-posterior surface contour from before.

``` r
beta_prior_grid %>% 
  rename(log_post = log_prior) %>% 
  mutate(N = 0) %>% 
  bind_rows(log_post_surface_result_01) %>% 
  group_by(N) %>% 
  mutate(max_log_post = max(log_post)) %>% 
  ungroup() %>% 
  mutate(log_post_2 = log_post - max_log_post) %>% 
  filter(N == 3) %>% 
  ggplot(mapping = aes(x = beta_1, y = beta_0)) +
  stat_contour(mapping = aes(z = log_post_2,
                             group = N),
               breaks = log(c(0.01/100, 0.01, 0.1, 0.5, 0.9)),
               size = 1.05,
               color = "black") +
  geom_point(data = isolate_subset_grid_01,
             size = 4.25,
             mapping = aes(color = log_post_2)) +
  geom_hline(yintercept = info_use_01$b0_mu,
             color = "dodgerblue") +
  geom_vline(xintercept = info_use_01$b1_mu,
             color = "dodgerblue") +
  geom_hline(yintercept = beta_0_true,
             color = "red", linetype = "dashed") +
  geom_vline(xintercept = beta_1_true,
             color = "red", linetype = "dashed") +
  coord_fixed(ratio = 1) +
  facet_wrap(~N, labeller = "label_both") +
  scale_color_viridis_c(guide = FALSE, option = "viridis",
                        limits = log(c(0.01/100, 1.0))) +
  labs(x = expression(beta[1]), y = expression(beta[0])) +
  theme_bw() +
  theme(legend.position = "top",
        axis.text = element_text(size = 12),
        axis.title = element_text(size = 12))
```

![](lecture_07_github_files/figure-markdown_github/viz_logpost_discrete_surface_01-1.png)

We went through this exercise of creating this "discrete" surface of a "relatively few" points because each ![\\beta\_{0}](https://latex.codecogs.com/png.latex?%5Cbeta_%7B0%7D "\beta_{0}") and ![\\beta\_{1}](https://latex.codecogs.com/png.latex?%5Cbeta_%7B1%7D "\beta_{1}") pair in the figure above corresponds to a separate linear predictor! We will now evaluate linear predictor at each grid point and plot the result as a function of the input. For context, the three specific observations for the ![N=3](https://latex.codecogs.com/png.latex?N%3D3 "N=3") case are included, along with the true linear predictor.

``` r
purrr::map2_dfr(isolate_subset_grid_01$beta_0,
                isolate_subset_grid_01$beta_1,
                calc_lin_predictor,
                xn = seq(-3, 3, length.out = 31)) %>% 
  ggplot(mapping = aes(x = x, y = mu)) +
  geom_line(mapping = aes(group = interaction(beta_0, beta_1))) +
  geom_point(data = train_df %>% 
               slice(1:3),
             mapping = aes(x = x, y = y),
             color = "red", size = 3.5) +
  geom_abline(slope = beta_1_true, intercept = beta_0_true,
              color = "red", linetype = "dashed", size = 1.05) +
  labs(x = "x", y = expression(mu)) +
  theme_bw()
```

![](lecture_07_github_files/figure-markdown_github/plot_many_lin_preds_01-1.png)

From the log-posterior surface, we know that not all of the ![\\beta\_{0}](https://latex.codecogs.com/png.latex?%5Cbeta_%7B0%7D "\beta_{0}") and ![\\beta\_{1}](https://latex.codecogs.com/png.latex?%5Cbeta_%7B1%7D "\beta_{1}") pairs are equally probable. In fact, most are highly implausible, and thus the lines corresponding to those unlikely pairs are themselves highly unlikely! To demonstrate this visually, set the transparency of each line to be equal to the posterior probability of the ![\\boldsymbol{\\beta}](https://latex.codecogs.com/png.latex?%5Cboldsymbol%7B%5Cbeta%7D "\boldsymbol{\beta}") pair the line corresponds to. All lines that have posterior probabilities of less than 1% have been set to a light grey color. Thus, only the lines that have 1% posterior probaility of occurring are visible. The code chunk below uses the `left_join()` function to merge two datasets together in order to access the `log_post_2` variable. **Which lines have definitely been ruled out as being highly unlikely?**

``` r
purrr::map2_dfr(isolate_subset_grid_01$beta_0,
                isolate_subset_grid_01$beta_1,
                calc_lin_predictor,
                xn = seq(-3, 3, length.out = 31)) %>% 
  left_join(isolate_subset_grid_01 %>% 
              select(beta_0, beta_1, log_post_2),
            by = c("beta_0", "beta_1")) %>% 
  ggplot(mapping = aes(x = x, y = mu)) +
  geom_line(mapping = aes(group = interaction(beta_0, beta_1),
                          alpha = exp(log_post_2),
                          color = exp(log_post_2) > 0.01)) +
  geom_point(data = train_df %>% 
               slice(1:3),
             mapping = aes(x = x, y = y),
             color = "red", size = 3.5) +
  scale_alpha_continuous(guide = FALSE) +
  scale_color_manual(guide = FALSE,
                     values = c("TRUE" = "black",
                                "FALSE" = "grey")) +
  labs(x = "x", y = expression(mu)) +
  theme_bw() +
  theme(legend.position = "top")
```

![](lecture_07_github_files/figure-markdown_github/plot_many_lin_preds_01_bb-1.png)

Let's drive this point home by directly sampling parameter pairs from the grid approximate posterior. In the code chunk below, 1000 ![\\beta\_{0}](https://latex.codecogs.com/png.latex?%5Cbeta_%7B0%7D "\beta_{0}") and ![\\beta\_{1}](https://latex.codecogs.com/png.latex?%5Cbeta_%7B1%7D "\beta_{1}") pairs are randomly selected from the posterior. Those random pairs are then used to calculate the posterior linear predictor with respect to the input. All 1000 lines are plotted in the figure generated from the code chunk below.

``` r
### set the posterior probability per beta pair
beta_post_grid_N03 <- beta_prior_grid %>% 
  rename(log_post = log_prior) %>% 
  mutate(N = 0) %>% 
  bind_rows(log_post_surface_result_01) %>% 
  group_by(N) %>% 
  mutate(max_log_post = max(log_post)) %>% 
  ungroup() %>% 
  mutate(log_post_2 = log_post - max_log_post) %>% 
  filter(N == 3)

### sample rows based on the posterior probability
set.seed(4110)
direct_sample_id <- sample(1:nrow(beta_post_grid_N03),
                           size = 1e3,
                           replace = TRUE,
                           prob = exp(beta_post_grid_N03$log_post_2))

beta_post_grid_N03_samples <- beta_post_grid_N03 %>% 
  slice(direct_sample_id)
```

``` r
purrr::map2_dfr(beta_post_grid_N03_samples$beta_0,
                beta_post_grid_N03_samples$beta_1,
                calc_lin_predictor,
                xn = seq(-3, 3, length.out = 31)) %>% 
  ggplot(mapping = aes(x = x, y = mu)) +
  geom_line(mapping = aes(group = interaction(beta_0, beta_1))) +
  geom_point(data = train_df %>% 
               slice(1:3),
             mapping = aes(x = x, y = y),
             color = "red", size = 3.5) +
  geom_abline(slope = beta_1_true, intercept = beta_0_true,
              color = "red", linetype = "dashed", size = 1.05) +
  labs(x = "x", y = expression(mu)) +
  theme_bw()
```

![](lecture_07_github_files/figure-markdown_github/viz_grid_approx_result_01_all_lines-1.png)

The figure above consists of a dark black band because all lines are fully opague. The important take away is that there are **zero** lines with negative slopes. Summarizing the 1000 lines gives us the posterior summaries on the linear predictor. The figure below represents the middle 95% uncertainty (credible) interval with the grey ribbon and the poster mean as the solid black line.

``` r
purrr::map2_dfr(beta_post_grid_N03_samples$beta_0,
                beta_post_grid_N03_samples$beta_1,
                calc_lin_predictor,
                xn = seq(-3, 3, length.out = 31)) %>% 
  ggplot(mapping = aes(x = x, y = mu)) +
  stat_summary(geom = "ribbon",
               fun.ymax = function(x){quantile(x, 0.95)},
               fun.ymin = function(x){quantile(x, 0.05)},
               fill = "grey50", alpha = 0.75) +
  stat_summary(geom = "line",
               fun.y = "mean",
               color = "black",
               size = 1.2) +
  geom_point(data = train_df %>% 
               slice(1:3),
             mapping = aes(x = x, y = y),
             color = "red", size = 3.5) +
  geom_abline(slope = beta_1_true, intercept = beta_0_true,
              color = "red", linetype = "dashed", size = 1.05) +
  labs(x = "x", y = expression(mu)) +
  theme_bw()
```

![](lecture_07_github_files/figure-markdown_github/viz_grid_approx_summary_01-1.png)

#### Diffuse prior

What if instead of the standard normal priors we used far more uncertain priors? What would you expect to occur? To see what happens, let's specify prior distributions still centered on 0, but now with standard deviations equal to 5 for both the intercept and the slope. In the code chunk below, after defining a new list, the prior surface is recalculated over the same grid of ![\\boldsymbol{\\beta}](https://latex.codecogs.com/png.latex?%5Cboldsymbol%7B%5Cbeta%7D "\boldsymbol{\beta}") as before.

``` r
hyper_diffuse_01 <- list(
  sigma = sigma_true,
  b0_mu = 0,
  b0_sd = 5,
  b1_mu = 0,
  b1_sd = 5
)

my_diffuse_prior_cov <- my_prior_cov <- matrix(c(5, 0, 0, 5), nrow = 2, byrow = TRUE)

beta_prior_diffuse_log_density <- mvtnorm::dmvnorm(
  as.matrix(beta_param_grid),
  mean = my_prior_mean,
  sigma = my_diffuse_prior_cov,
  log = TRUE)
```

Visualizing the diffuse prior surface reveals a flatter surface in both parameter directions.

``` r
beta_param_grid %>% 
  mutate(log_prior = beta_prior_diffuse_log_density) %>% 
  mutate(log_prior_2 = log_prior - max(log_prior)) %>% 
  ggplot(mapping = aes(x = beta_1, y = beta_0)) +
  geom_raster(mapping = aes(fill = log_prior_2)) +
  stat_contour(mapping = aes(z = log_prior_2),
               breaks = log(c(0.01/100, 0.01, 0.1, 0.5, 0.9)),
               size = 2.2,
               color = "black") +
  geom_hline(yintercept = beta_0_true,
             color = "red", linetype = "dashed") +
  geom_vline(xintercept = beta_1_true,
             color = "red", linetype = "dashed") +
  coord_fixed(ratio = 1) +
  scale_fill_viridis_c(guide = FALSE, option = "viridis",
                       limits = log(c(0.01/100, 1.0))) +
  labs(x = expression(beta[1]), y = expression(beta[0])) +
  theme_bw() +
  theme(legend.position = "top",
        axis.text = element_text(size = 12),
        axis.title = element_text(size = 12))
```

![](lecture_07_github_files/figure-markdown_github/viz_diffuse_prior_surface-1.png)

Let's now re-evaluate the log-posterior surface by sequentially adding points. We will add up to the first 10 points, just as we did before.

``` r
log_post_surface_diffuse_01 <- purrr::map_dfr(1:10,
                                              manage_log_post_01,
                                              avail_data = demo_df,
                                              my_settings = hyper_diffuse_01,
                                              grid_use = beta_param_grid)
```

Visualize the log-posterior surface contours as we add each data point, starting from the diffuse prior.

``` r
beta_param_grid %>% 
  mutate(log_post = beta_prior_diffuse_log_density) %>% 
  mutate(N = 0) %>% 
  bind_rows(log_post_surface_diffuse_01) %>% 
  group_by(N) %>% 
  mutate(max_log_post = max(log_post)) %>% 
  ungroup() %>% 
  mutate(log_post_2 = log_post - max_log_post) %>% 
  ggplot(mapping = aes(x = beta_1, y = beta_0)) +
  stat_contour(mapping = aes(z = log_post_2,
                             group = N),
               breaks = log(c(0.01/100, 0.01, 0.1, 0.5, 0.9)),
               size = 1.05,
               color = "black") +
  geom_hline(yintercept = beta_0_true,
             color = "red", linetype = "dashed") +
  geom_vline(xintercept = beta_1_true,
             color = "red", linetype = "dashed") +
  coord_fixed(ratio = 1) +
  facet_wrap(~N, labeller = "label_both") +
  scale_fill_viridis_c(guide = FALSE, option = "viridis",
                       limits = log(c(0.01/100, 1.0))) +
  labs(x = expression(beta[1]), y = expression(beta[0])) +
  theme_bw() +
  theme(legend.position = "top",
        axis.text = element_text(size = 12),
        axis.title = element_text(size = 12))
```

![](lecture_07_github_files/figure-markdown_github/viz_seq_grid_approx_diffuse_01-1.png)

#### Laplace approximation

Now that we know what the log-posterior surface looks like, let's apply the Laplace approximation. We saw how to setup the Laplace apprxomiation last week. We will simply reuse the `my_laplace()` we discussed in lecture last week.

``` r
my_laplace <- function(start_guess, logpost_func, ...)
{
  # code adapted from the `LearnBayes`` function `laplace()`
  fit <- optim(start_guess,
               logpost_func,
               gr = NULL,
               ...,
               method = "BFGS",
               hessian = TRUE,
               control = list(fnscale = -1, maxit = 1001))
  
  mode <- fit$par
  h <- -solve(fit$hessian)
  p <- length(mode)
  int <- p/2 * log(2 * pi) + 0.5 * log(det(h)) + logpost_func(mode, ...)
  list(mode = mode,
       var_matrix = h,
       log_evidence = int,
       converge = ifelse(fit$convergence == 0,
                         "YES", 
                         "NO"),
       iter_counts = fit$counts[1])
}
```

Let's wrap the `my_laplace()` function in a function which allows us to visualize the MVN approximate posterior over a defined grid of parameter values. This way we can compare the Laplace approximation to the true log-posterior surface.

``` r
viz_mvn_approx_post <- function(num_obs, avail_data, my_settings, grid_use, init_guess, logpost_func)
{
  # add in the observations correctly
  my_settings$xobs <- avail_data$x[1:num_obs]
  my_settings$yobs <- avail_data$y[1:num_obs]
  
  # execute the laplace approximation
  laplace_result <- my_laplace(init_guess, logpost_func, my_settings)
  
  # evaluate the MVN approx posterior log-density
  approx_logpost <- mvtnorm::dmvnorm(as.matrix(grid_use),
                                     mean = laplace_result$mode,
                                     sigma = laplace_result$var_matrix,
                                     log = TRUE)
  
  # package everything together
  grid_use %>% 
    mutate(log_post = approx_logpost) %>% 
    mutate(N = num_obs,
           type = "Laplace")
}
```

Sequentially apply the Laplace approximation up to and including the first 5 observations. We will continue to use the diffuse prior on the ![\\boldsymbol{\\beta}](https://latex.codecogs.com/png.latex?%5Cboldsymbol%7B%5Cbeta%7D "\boldsymbol{\beta}"). Set the initial guess to be ![\\boldsymbol{\\beta}\_{\\mathrm{init}}=\\{-1, -1\\}](https://latex.codecogs.com/png.latex?%5Cboldsymbol%7B%5Cbeta%7D_%7B%5Cmathrm%7Binit%7D%7D%3D%5C%7B-1%2C%20-1%5C%7D "\boldsymbol{\beta}_{\mathrm{init}}=\{-1, -1\}").

``` r
approx_log_post_surface_result_01 <- purrr::map_dfr(1:5,
                                                    viz_mvn_approx_post,
                                                    avail_data = demo_df,
                                                    my_settings = hyper_diffuse_01,
                                                    grid_use = beta_param_grid,
                                                    init_guess = c(-1, -1),
                                                    logpost_func = lm_logpost_01)
```

Let's now compare the approximate log-posterior with the true log-posterior surface. In the figure below, the black contours are the true log-posterior contours from the Grid approximation and the dashed red contours are the MVN approximate log-posterior contours. If you remember, these are the same colors we used last week when comparing the true and Laplace approximate log-posteriors. Due to this color scheme, the *true* ![\\boldsymbol{\\beta}](https://latex.codecogs.com/png.latex?%5Cboldsymbol%7B%5Cbeta%7D "\boldsymbol{\beta}") values are denoted by navy blue dashed lines, instead of dashed red lines. As shown in the figure below, the true log-posterior and the MVN approximate log-posterior are the same! **Why should we have anticipated the result shown in the figure below?**

``` r
beta_param_grid %>% 
  mutate(log_post = beta_prior_diffuse_log_density) %>% 
  mutate(N = 0) %>% 
  bind_rows(log_post_surface_diffuse_01) %>% 
  mutate(type = "Grid") %>% 
  filter(N < 6) %>% 
  bind_rows(approx_log_post_surface_result_01) %>% 
  group_by(N, type) %>% 
  mutate(max_log_post = max(log_post)) %>% 
  ungroup() %>% 
  mutate(log_post_2 = log_post - max_log_post) %>% 
  ggplot(mapping = aes(x = beta_1, y = beta_0)) +
  stat_contour(mapping = aes(z = log_post_2,
                             group = interaction(N, type),
                             color = type,
                             linetype = type),
               breaks = log(c(0.01/100, 0.01, 0.1, 0.5, 0.9)),
               size = 1.05) +
  geom_hline(yintercept = beta_0_true,
             color = "navyblue", linetype = "dashed") +
  geom_vline(xintercept = beta_1_true,
             color = "navyblue", linetype = "dashed") +
  coord_fixed(ratio = 1) +
  facet_wrap(~N, labeller = "label_both") +
  scale_color_manual("Method",
                     values = c("Grid" = "black",
                                "Laplace" = "red")) +
  scale_linetype_manual("Method",
                        values = c("Grid" = "solid",
                                   "Laplace" = "dashed")) +
  labs(x = expression(beta[1]), y = expression(beta[0])) +
  theme_bw() +
  theme(axis.text = element_text(size = 12),
        axis.title = element_text(size = 12))
```

![](lecture_07_github_files/figure-markdown_github/compare_grid_laplace_contours_01-1.png)

### Derivation

Although we have several techniques to solve this problem, let's step into the math to see how to derive the result. We will derive the posterior via the Laplace approximation, and since the posterior is a MVN (as we saw above), the Laplace approximation will therefore yield the exact result.

We shall first consider the case of the infinitely diffuse prior on ![\\boldsymbol{\\beta}](https://latex.codecogs.com/png.latex?%5Cboldsymbol%7B%5Cbeta%7D "\boldsymbol{\beta}"). Under this assumption, the log-posterior is essentially the same as the log-likelihood since the contribution of the log-prior is negligible. Write out the terms in the log-likelihood which involve the unknown ![\\boldsymbol{\\beta}](https://latex.codecogs.com/png.latex?%5Cboldsymbol%7B%5Cbeta%7D "\boldsymbol{\beta}") parameters:

![ 
\\log\\left\[ p\\left( \\boldsymbol{\\beta} \\mid \\mathbf{x}, \\mathbf{y}, \\sigma \\right) \\right\] \\propto \\log\\left\[ p\\left( \\mathbf{y} \\mid \\mathbf{x},\\boldsymbol{\\beta},\\sigma \\right)\\right\] + \\log\\left\[ p\\left( \\boldsymbol{\\beta}\\right) \\right\] \\approx \\log\\left\[ p\\left( \\mathbf{y} \\mid \\mathbf{x},\\boldsymbol{\\beta},\\sigma \\right)\\right\] \\\\ \\log\\left\[ p\\left( \\boldsymbol{\\beta} \\mid \\mathbf{x}, \\mathbf{y}, \\sigma \\right) \\right\] \\approx -\\frac{1}{2\\sigma^{2}} \\sum\_{n=1}^{N} \\left( \\left(y\_{n} - \\mu\_{n} \\right)^2 \\right)
](https://latex.codecogs.com/png.latex?%20%0A%5Clog%5Cleft%5B%20p%5Cleft%28%20%5Cboldsymbol%7B%5Cbeta%7D%20%5Cmid%20%5Cmathbf%7Bx%7D%2C%20%5Cmathbf%7By%7D%2C%20%5Csigma%20%5Cright%29%20%5Cright%5D%20%5Cpropto%20%5Clog%5Cleft%5B%20p%5Cleft%28%20%5Cmathbf%7By%7D%20%5Cmid%20%5Cmathbf%7Bx%7D%2C%5Cboldsymbol%7B%5Cbeta%7D%2C%5Csigma%20%5Cright%29%5Cright%5D%20%2B%20%5Clog%5Cleft%5B%20p%5Cleft%28%20%5Cboldsymbol%7B%5Cbeta%7D%5Cright%29%20%5Cright%5D%20%5Capprox%20%5Clog%5Cleft%5B%20p%5Cleft%28%20%5Cmathbf%7By%7D%20%5Cmid%20%5Cmathbf%7Bx%7D%2C%5Cboldsymbol%7B%5Cbeta%7D%2C%5Csigma%20%5Cright%29%5Cright%5D%20%5C%5C%20%5Clog%5Cleft%5B%20p%5Cleft%28%20%5Cboldsymbol%7B%5Cbeta%7D%20%5Cmid%20%5Cmathbf%7Bx%7D%2C%20%5Cmathbf%7By%7D%2C%20%5Csigma%20%5Cright%29%20%5Cright%5D%20%5Capprox%20-%5Cfrac%7B1%7D%7B2%5Csigma%5E%7B2%7D%7D%20%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20%5Cleft%28%20%5Cleft%28y_%7Bn%7D%20-%20%5Cmu_%7Bn%7D%20%5Cright%29%5E2%20%5Cright%29%0A " 
\log\left[ p\left( \boldsymbol{\beta} \mid \mathbf{x}, \mathbf{y}, \sigma \right) \right] \propto \log\left[ p\left( \mathbf{y} \mid \mathbf{x},\boldsymbol{\beta},\sigma \right)\right] + \log\left[ p\left( \boldsymbol{\beta}\right) \right] \approx \log\left[ p\left( \mathbf{y} \mid \mathbf{x},\boldsymbol{\beta},\sigma \right)\right] \\ \log\left[ p\left( \boldsymbol{\beta} \mid \mathbf{x}, \mathbf{y}, \sigma \right) \right] \approx -\frac{1}{2\sigma^{2}} \sum_{n=1}^{N} \left( \left(y_{n} - \mu_{n} \right)^2 \right)
")

**What does the above expression remind you of?**

If we define the residual between the linear predictor and the observation as:

![ 
\\epsilon\_{n} = y\_{n} - \\mu\_{n}
](https://latex.codecogs.com/png.latex?%20%0A%5Cepsilon_%7Bn%7D%20%3D%20y_%7Bn%7D%20-%20%5Cmu_%7Bn%7D%0A " 
\epsilon_{n} = y_{n} - \mu_{n}
")

The summation term is just the **residual sum of squares** or the **sum of squared errors**:

![ 
\\sum\_{n=1}^{N} \\left( \\epsilon\_{n}^{2} \\right)
](https://latex.codecogs.com/png.latex?%20%0A%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20%5Cleft%28%20%5Cepsilon_%7Bn%7D%5E%7B2%7D%20%5Cright%29%0A " 
\sum_{n=1}^{N} \left( \epsilon_{n}^{2} \right)
")

 Thus, under the assumption of an infinitely diffuse prior, maximizing the log-posterior will be equivalent to minimizing the the squared error!

#### Matrix-vector notation

Before substituting the deterministic function for the linear predictor, ![\\mu\_{n}](https://latex.codecogs.com/png.latex?%5Cmu_%7Bn%7D "\mu_{n}"), we will make a slight change to the syntax we have used up to this point. Let's create a "fake" variable, ![x\_0](https://latex.codecogs.com/png.latex?x_0 "x_0"), and redefine out input as ![x\_1](https://latex.codecogs.com/png.latex?x_1 "x_1"). Our fake variable will multiple the intercept, ![\\beta\_{0}](https://latex.codecogs.com/png.latex?%5Cbeta_%7B0%7D "\beta_{0}"), and thus will always be equal to 1. The linear predictor expression is then rewritten as:

![ 
\\mu\_{n} = \\beta\_{0} x\_{n,0} + \\beta\_{1} x\_{n,1}
](https://latex.codecogs.com/png.latex?%20%0A%5Cmu_%7Bn%7D%20%3D%20%5Cbeta_%7B0%7D%20x_%7Bn%2C0%7D%20%2B%20%5Cbeta_%7B1%7D%20x_%7Bn%2C1%7D%0A " 
\mu_{n} = \beta_{0} x_{n,0} + \beta_{1} x_{n,1}
")

We can therefore generalize the linear predictor expression for a case ![D](https://latex.codecogs.com/png.latex?D "D") inputs by writing:

![ 
\\mu\_{n} = \\sum\_{d=0}^{D} \\left( \\beta\_{d} x\_{n,d} \\right)
](https://latex.codecogs.com/png.latex?%20%0A%5Cmu_%7Bn%7D%20%3D%20%5Csum_%7Bd%3D0%7D%5E%7BD%7D%20%5Cleft%28%20%5Cbeta_%7Bd%7D%20x_%7Bn%2Cd%7D%20%5Cright%29%0A " 
\mu_{n} = \sum_{d=0}^{D} \left( \beta_{d} x_{n,d} \right)
")

Notice that we are writing the ![n](https://latex.codecogs.com/png.latex?n "n")-th observation of the ![d](https://latex.codecogs.com/png.latex?d "d")-th variable with the subscript ![n,d](https://latex.codecogs.com/png.latex?n%2Cd "n,d"). If we organize the ![D+1](https://latex.codecogs.com/png.latex?D%2B1 "D+1") unknown parameters into a column vector ![\\boldsymbol{\\beta}](https://latex.codecogs.com/png.latex?%5Cboldsymbol%7B%5Cbeta%7D "\boldsymbol{\beta}"), we can rewrite the ![n](https://latex.codecogs.com/png.latex?n "n")-th linear predictor as the dot product of ![\\left(1, D+1 \\right)](https://latex.codecogs.com/png.latex?%5Cleft%281%2C%20D%2B1%20%5Cright%29 "\left(1, D+1 \right)") row vector ![\\mathrm{x}\_{n,:}](https://latex.codecogs.com/png.latex?%5Cmathrm%7Bx%7D_%7Bn%2C%3A%7D "\mathrm{x}_{n,:}") and ![\\boldsymbol{\\beta}](https://latex.codecogs.com/png.latex?%5Cboldsymbol%7B%5Cbeta%7D "\boldsymbol{\beta}").

![ 
\\mu\_{n} = \\mathrm{x}\_{n,:}\\boldsymbol{\\beta}
](https://latex.codecogs.com/png.latex?%20%0A%5Cmu_%7Bn%7D%20%3D%20%5Cmathrm%7Bx%7D_%7Bn%2C%3A%7D%5Cboldsymbol%7B%5Cbeta%7D%0A " 
\mu_{n} = \mathrm{x}_{n,:}\boldsymbol{\beta}
")

 We can organize the ![D+1](https://latex.codecogs.com/png.latex?D%2B1 "D+1") separate vectors, each of length ![N](https://latex.codecogs.com/png.latex?N "N"), into the ![\\left(N, D+1\\right)](https://latex.codecogs.com/png.latex?%5Cleft%28N%2C%20D%2B1%5Cright%29 "\left(N, D+1\right)") **design** matrix ![\\mathbf{X}](https://latex.codecogs.com/png.latex?%5Cmathbf%7BX%7D "\mathbf{X}"). For our specific example with 1 input, the design matrix is:

![ 
\\mathbf{X} = \\left\[\\begin{array}
{rr}
x\_{1,0} & x\_{1,1} \\\\
\\vdots & \\vdots \\\\
x\_{N,0} & x\_{N,1} 
\\end{array}\\right\] = \\left\[\\begin{array}
{rr}
1 & x\_{1,1} \\\\
\\vdots & \\vdots \\\\
1 & x\_{N,1} 
\\end{array}\\right\]
](https://latex.codecogs.com/png.latex?%20%0A%5Cmathbf%7BX%7D%20%3D%20%5Cleft%5B%5Cbegin%7Barray%7D%0A%7Brr%7D%0Ax_%7B1%2C0%7D%20%26%20x_%7B1%2C1%7D%20%5C%5C%0A%5Cvdots%20%26%20%5Cvdots%20%5C%5C%0Ax_%7BN%2C0%7D%20%26%20x_%7BN%2C1%7D%20%0A%5Cend%7Barray%7D%5Cright%5D%20%3D%20%5Cleft%5B%5Cbegin%7Barray%7D%0A%7Brr%7D%0A1%20%26%20x_%7B1%2C1%7D%20%5C%5C%0A%5Cvdots%20%26%20%5Cvdots%20%5C%5C%0A1%20%26%20x_%7BN%2C1%7D%20%0A%5Cend%7Barray%7D%5Cright%5D%0A " 
\mathbf{X} = \left[\begin{array}
{rr}
x_{1,0} & x_{1,1} \\
\vdots & \vdots \\
x_{N,0} & x_{N,1} 
\end{array}\right] = \left[\begin{array}
{rr}
1 & x_{1,1} \\
\vdots & \vdots \\
1 & x_{N,1} 
\end{array}\right]
")

In general, with ![D](https://latex.codecogs.com/png.latex?D "D") variables, the design matrix can be written as:

![ 
\\mathbf{X} = \\left\[\\begin{array}
{rrrr}
x\_{1,0} & x\_{1,1} & \\ldots & x\_{1,D} \\\\
\\vdots & \\vdots & \\ddots & \\vdots\\\\
x\_{N,0} & x\_{N,1} & \\ldots & x\_{N,D} 
\\end{array}\\right\] = \\left\[\\begin{array}
{rrrr}
1 & x\_{1,1} & \\ldots & x\_{1,D} \\\\
\\vdots & \\vdots & \\ddots & \\vdots\\\\
1 & x\_{N,1} & \\ldots & x\_{N,D} 
\\end{array}\\right\]
](https://latex.codecogs.com/png.latex?%20%0A%5Cmathbf%7BX%7D%20%3D%20%5Cleft%5B%5Cbegin%7Barray%7D%0A%7Brrrr%7D%0Ax_%7B1%2C0%7D%20%26%20x_%7B1%2C1%7D%20%26%20%5Cldots%20%26%20x_%7B1%2CD%7D%20%5C%5C%0A%5Cvdots%20%26%20%5Cvdots%20%26%20%5Cddots%20%26%20%5Cvdots%5C%5C%0Ax_%7BN%2C0%7D%20%26%20x_%7BN%2C1%7D%20%26%20%5Cldots%20%26%20x_%7BN%2CD%7D%20%0A%5Cend%7Barray%7D%5Cright%5D%20%3D%20%5Cleft%5B%5Cbegin%7Barray%7D%0A%7Brrrr%7D%0A1%20%26%20x_%7B1%2C1%7D%20%26%20%5Cldots%20%26%20x_%7B1%2CD%7D%20%5C%5C%0A%5Cvdots%20%26%20%5Cvdots%20%26%20%5Cddots%20%26%20%5Cvdots%5C%5C%0A1%20%26%20x_%7BN%2C1%7D%20%26%20%5Cldots%20%26%20x_%7BN%2CD%7D%20%0A%5Cend%7Barray%7D%5Cright%5D%0A " 
\mathbf{X} = \left[\begin{array}
{rrrr}
x_{1,0} & x_{1,1} & \ldots & x_{1,D} \\
\vdots & \vdots & \ddots & \vdots\\
x_{N,0} & x_{N,1} & \ldots & x_{N,D} 
\end{array}\right] = \left[\begin{array}
{rrrr}
1 & x_{1,1} & \ldots & x_{1,D} \\
\vdots & \vdots & \ddots & \vdots\\
1 & x_{N,1} & \ldots & x_{N,D} 
\end{array}\right]
")

The ![N](https://latex.codecogs.com/png.latex?N "N") length linear predictor vector, ![\\boldsymbol{\\mu}](https://latex.codecogs.com/png.latex?%5Cboldsymbol%7B%5Cmu%7D "\boldsymbol{\mu}"), can then be written in terms of the design matrix and a column vector of parameters, ![\\boldsymbol{\\beta}](https://latex.codecogs.com/png.latex?%5Cboldsymbol%7B%5Cbeta%7D "\boldsymbol{\beta}"), of size ![\\left( D+1, 1\\right)](https://latex.codecogs.com/png.latex?%5Cleft%28%20D%2B1%2C%201%5Cright%29 "\left( D+1, 1\right)"). The expression below is created by essentially "stacking" each of the ![n](https://latex.codecogs.com/png.latex?n "n") separate calculations together. The matrix/vector aglebra simply takes care of the "stacking" for us.

![ 
\\boldsymbol{\\mu} = \\mathbf{X}\\boldsymbol{\\beta}
](https://latex.codecogs.com/png.latex?%20%0A%5Cboldsymbol%7B%5Cmu%7D%20%3D%20%5Cmathbf%7BX%7D%5Cboldsymbol%7B%5Cbeta%7D%0A " 
\boldsymbol{\mu} = \mathbf{X}\boldsymbol{\beta}
")

Using matrix/vector notation we can rewite the log-posterior under the assumption of a diffuse prior as:

![ 
-\\frac{1}{2\\sigma^{2}} \\sum\_{n=1}^{N} \\left( \\left(y\_{n} - \\mu\_{n} \\right)^2 \\right) = -\\frac{1}{2\\sigma^2}\\sum\_{n=1}^{N} \\left( \\left( y\_{n} - \\mathrm{x}\_{n,:} \\boldsymbol{\\beta} \\right)^{2} \\right) = -\\frac{1}{2\\sigma^2}\\left( \\mathbf{y} - \\mathbf{X}\\boldsymbol{\\beta} \\right)^{T} \\left( \\mathbf{y} - \\mathbf{X}\\boldsymbol{\\beta} \\right) 
](https://latex.codecogs.com/png.latex?%20%0A-%5Cfrac%7B1%7D%7B2%5Csigma%5E%7B2%7D%7D%20%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20%5Cleft%28%20%5Cleft%28y_%7Bn%7D%20-%20%5Cmu_%7Bn%7D%20%5Cright%29%5E2%20%5Cright%29%20%3D%20-%5Cfrac%7B1%7D%7B2%5Csigma%5E2%7D%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20%5Cleft%28%20%5Cleft%28%20y_%7Bn%7D%20-%20%5Cmathrm%7Bx%7D_%7Bn%2C%3A%7D%20%5Cboldsymbol%7B%5Cbeta%7D%20%5Cright%29%5E%7B2%7D%20%5Cright%29%20%3D%20-%5Cfrac%7B1%7D%7B2%5Csigma%5E2%7D%5Cleft%28%20%5Cmathbf%7By%7D%20-%20%5Cmathbf%7BX%7D%5Cboldsymbol%7B%5Cbeta%7D%20%5Cright%29%5E%7BT%7D%20%5Cleft%28%20%5Cmathbf%7By%7D%20-%20%5Cmathbf%7BX%7D%5Cboldsymbol%7B%5Cbeta%7D%20%5Cright%29%20%0A " 
-\frac{1}{2\sigma^{2}} \sum_{n=1}^{N} \left( \left(y_{n} - \mu_{n} \right)^2 \right) = -\frac{1}{2\sigma^2}\sum_{n=1}^{N} \left( \left( y_{n} - \mathrm{x}_{n,:} \boldsymbol{\beta} \right)^{2} \right) = -\frac{1}{2\sigma^2}\left( \mathbf{y} - \mathbf{X}\boldsymbol{\beta} \right)^{T} \left( \mathbf{y} - \mathbf{X}\boldsymbol{\beta} \right) 
")

For notational simplicity, we will define the above expression as ![L](https://latex.codecogs.com/png.latex?L "L").

#### Gradient

The Laplace approximation requires finding the posterior mode, or *max a posteriori* estimate. We will therefore need to evaluate the partial derivative of ![L](https://latex.codecogs.com/png.latex?L "L") with respect to each unknown parameter. Let's continue to work in general terms by taking the partial derivative of ![L](https://latex.codecogs.com/png.latex?L "L") with respect to the ![\\beta\_{d}](https://latex.codecogs.com/png.latex?%5Cbeta_%7Bd%7D "\beta_{d}") parameter:

![ 
\\frac{\\partial L}{\\partial \\beta\_{d}} = -\\frac{1}{2\\sigma^2}\\sum\_{n=1}^{N}\\left( \\frac{\\partial}{\\partial \\beta\_{d}} \\left( \\left( y\_n - \\mathbf{x}\_{n,:} \\boldsymbol{\\beta} \\right)^2 \\right) \\right)
](https://latex.codecogs.com/png.latex?%20%0A%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20%5Cbeta_%7Bd%7D%7D%20%3D%20-%5Cfrac%7B1%7D%7B2%5Csigma%5E2%7D%5Csum_%7Bn%3D1%7D%5E%7BN%7D%5Cleft%28%20%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20%5Cbeta_%7Bd%7D%7D%20%5Cleft%28%20%5Cleft%28%20y_n%20-%20%5Cmathbf%7Bx%7D_%7Bn%2C%3A%7D%20%5Cboldsymbol%7B%5Cbeta%7D%20%5Cright%29%5E2%20%5Cright%29%20%5Cright%29%0A " 
\frac{\partial L}{\partial \beta_{d}} = -\frac{1}{2\sigma^2}\sum_{n=1}^{N}\left( \frac{\partial}{\partial \beta_{d}} \left( \left( y_n - \mathbf{x}_{n,:} \boldsymbol{\beta} \right)^2 \right) \right)
")

The partial derivative of the term within the summation is:

![ 
\\frac{\\partial}{\\partial \\beta\_{d}} \\left( \\left( y\_n - \\mathbf{x}\_{n,:} \\boldsymbol{\\beta} \\right)^2 \\right) = 2 \\left( y\_{n} - \\mathbf{x}\_{n,:} \\boldsymbol{\\beta} \\right) \\cdot \\left(-x\_{n,d} \\right)
](https://latex.codecogs.com/png.latex?%20%0A%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20%5Cbeta_%7Bd%7D%7D%20%5Cleft%28%20%5Cleft%28%20y_n%20-%20%5Cmathbf%7Bx%7D_%7Bn%2C%3A%7D%20%5Cboldsymbol%7B%5Cbeta%7D%20%5Cright%29%5E2%20%5Cright%29%20%3D%202%20%5Cleft%28%20y_%7Bn%7D%20-%20%5Cmathbf%7Bx%7D_%7Bn%2C%3A%7D%20%5Cboldsymbol%7B%5Cbeta%7D%20%5Cright%29%20%5Ccdot%20%5Cleft%28-x_%7Bn%2Cd%7D%20%5Cright%29%0A " 
\frac{\partial}{\partial \beta_{d}} \left( \left( y_n - \mathbf{x}_{n,:} \boldsymbol{\beta} \right)^2 \right) = 2 \left( y_{n} - \mathbf{x}_{n,:} \boldsymbol{\beta} \right) \cdot \left(-x_{n,d} \right)
")

Substituting this expression back into the summation and simplifying gives:

![ 
\\frac{\\partial L}{\\partial \\beta\_{d}} = \\frac{1}{\\sigma^2} \\sum\_{n=1}^{N} \\left( y\_{n} x\_{n,d} - \\left(\\mathbf{x}\_{n,:}\\boldsymbol{\\beta} \\right) \\cdot x\_{n,d} \\right)
](https://latex.codecogs.com/png.latex?%20%0A%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20%5Cbeta_%7Bd%7D%7D%20%3D%20%5Cfrac%7B1%7D%7B%5Csigma%5E2%7D%20%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20%5Cleft%28%20y_%7Bn%7D%20x_%7Bn%2Cd%7D%20-%20%5Cleft%28%5Cmathbf%7Bx%7D_%7Bn%2C%3A%7D%5Cboldsymbol%7B%5Cbeta%7D%20%5Cright%29%20%5Ccdot%20x_%7Bn%2Cd%7D%20%5Cright%29%0A " 
\frac{\partial L}{\partial \beta_{d}} = \frac{1}{\sigma^2} \sum_{n=1}^{N} \left( y_{n} x_{n,d} - \left(\mathbf{x}_{n,:}\boldsymbol{\beta} \right) \cdot x_{n,d} \right)
")

The summation can be distributed to each of the terms:

![ 
\\frac{\\partial L}{\\partial \\beta\_{d}} = \\frac{1}{\\sigma^2} \\left( \\sum\_{n=1}^{N} \\left(y\_n x\_{n,d} \\right) - \\sum\_{n=1}^{N} \\left( \\left( \\mathbf{x}\_{n,:} \\boldsymbol{\\beta} \\right) \\cdot x\_{n,d} \\right) \\right)
](https://latex.codecogs.com/png.latex?%20%0A%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20%5Cbeta_%7Bd%7D%7D%20%3D%20%5Cfrac%7B1%7D%7B%5Csigma%5E2%7D%20%5Cleft%28%20%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20%5Cleft%28y_n%20x_%7Bn%2Cd%7D%20%5Cright%29%20-%20%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20%5Cleft%28%20%5Cleft%28%20%5Cmathbf%7Bx%7D_%7Bn%2C%3A%7D%20%5Cboldsymbol%7B%5Cbeta%7D%20%5Cright%29%20%5Ccdot%20x_%7Bn%2Cd%7D%20%5Cright%29%20%5Cright%29%0A " 
\frac{\partial L}{\partial \beta_{d}} = \frac{1}{\sigma^2} \left( \sum_{n=1}^{N} \left(y_n x_{n,d} \right) - \sum_{n=1}^{N} \left( \left( \mathbf{x}_{n,:} \boldsymbol{\beta} \right) \cdot x_{n,d} \right) \right)
")

The first summation series is the dot product of the ![\\mathbf{y}](https://latex.codecogs.com/png.latex?%5Cmathbf%7By%7D "\mathbf{y}") vector with all vector of observations for the ![d](https://latex.codecogs.com/png.latex?d "d")-th input variable, ![\\mathbf{x}\_{:,d}](https://latex.codecogs.com/png.latex?%5Cmathbf%7Bx%7D_%7B%3A%2Cd%7D "\mathbf{x}_{:,d}"):

![ 
\\sum\_{n=1}^{N} \\left(y\_n x\_{n,d} \\right) = \\mathrm{x}\_{:,d}^T \\mathbf{y}
](https://latex.codecogs.com/png.latex?%20%0A%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20%5Cleft%28y_n%20x_%7Bn%2Cd%7D%20%5Cright%29%20%3D%20%5Cmathrm%7Bx%7D_%7B%3A%2Cd%7D%5ET%20%5Cmathbf%7By%7D%0A " 
\sum_{n=1}^{N} \left(y_n x_{n,d} \right) = \mathrm{x}_{:,d}^T \mathbf{y}
")

The expression above produces a scalar number, associated with the ![d](https://latex.codecogs.com/png.latex?d "d")-th parameter. We can repeat that calculation for all ![D+1](https://latex.codecogs.com/png.latex?D%2B1 "D+1") parameters, or we can use matrix algebra to repeat the operation for us. Instead of multiplying the transpose of the ![d](https://latex.codecogs.com/png.latex?d "d")-th column of the design matrix by the vector ![\\mathbf{y}](https://latex.codecogs.com/png.latex?%5Cmathbf%7By%7D "\mathbf{y}"), we multiply the transpose of the entire design matrix by the vector ![\\mathbf{y}](https://latex.codecogs.com/png.latex?%5Cmathbf%7By%7D "\mathbf{y}"):

![ 
\\mathbf{X}^T \\mathbf{y}
](https://latex.codecogs.com/png.latex?%20%0A%5Cmathbf%7BX%7D%5ET%20%5Cmathbf%7By%7D%0A " 
\mathbf{X}^T \mathbf{y}
")

The second summation series in the derivative is a little more challenging. It is repeated below for conenience.

![ 
\\sum\_{n=1}^{N} \\left( \\left( \\mathbf{x}\_{n,:} \\boldsymbol{\\beta} \\right) \\cdot x\_{n,d} \\right)
](https://latex.codecogs.com/png.latex?%20%0A%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20%5Cleft%28%20%5Cleft%28%20%5Cmathbf%7Bx%7D_%7Bn%2C%3A%7D%20%5Cboldsymbol%7B%5Cbeta%7D%20%5Cright%29%20%5Ccdot%20x_%7Bn%2Cd%7D%20%5Cright%29%0A " 
\sum_{n=1}^{N} \left( \left( \mathbf{x}_{n,:} \boldsymbol{\beta} \right) \cdot x_{n,d} \right)
")

First, consider that we are summing over all ![N](https://latex.codecogs.com/png.latex?N "N") observations. Second, each observation of the ![d](https://latex.codecogs.com/png.latex?d "d")-input is multiplied by the linear predictor associated with the ![n](https://latex.codecogs.com/png.latex?n "n")-th observation. We are therefore multiplying the ![n](https://latex.codecogs.com/png.latex?n "n")-th observation of the ![d](https://latex.codecogs.com/png.latex?d "d")-th input variable by the ![n](https://latex.codecogs.com/png.latex?n "n")-th observation of all ![D+1](https://latex.codecogs.com/png.latex?D%2B1 "D+1") input variables. We then take the dot product with the parameter vector. To make this more clear, let's rearrange the above expression to be:

![ 
\\sum\_{n=1}^{N} \\left( \\left( x\_{n,d} \\cdot \\mathbf{x}\_{n,:} \\right) \\boldsymbol{\\beta} \\right)
](https://latex.codecogs.com/png.latex?%20%0A%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20%5Cleft%28%20%5Cleft%28%20x_%7Bn%2Cd%7D%20%5Ccdot%20%5Cmathbf%7Bx%7D_%7Bn%2C%3A%7D%20%5Cright%29%20%5Cboldsymbol%7B%5Cbeta%7D%20%5Cright%29%0A " 
\sum_{n=1}^{N} \left( \left( x_{n,d} \cdot \mathbf{x}_{n,:} \right) \boldsymbol{\beta} \right)
")

To extend this expression to all ![D+1](https://latex.codecogs.com/png.latex?D%2B1 "D+1") parameters, we need to define the **sum of squares** matrix. The name comes from the fact that we are squaring many terms and then summing over all observations:

![ 
\\sum\_{n=1}^{N} \\left( \\mathbf{x}\_{n,:}^{T} \\mathbf{x}\_{n,:} \\right) = \\sum\_{n=1}^{N} \\left( \\left\[\\begin{array}
{rrrr}
x\_{n,0}^{2} & x\_{n,0}x\_{n,1} & \\ldots & x\_{n,0}x\_{n,D} \\\\
\\vdots & \\vdots & \\ddots & \\vdots\\\\
x\_{n,D}x\_{n,0} & x\_{n,D}x\_{n,1} & \\ldots & x\_{n,D}^2 
\\end{array}\\right\] \\right)
](https://latex.codecogs.com/png.latex?%20%0A%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20%5Cleft%28%20%5Cmathbf%7Bx%7D_%7Bn%2C%3A%7D%5E%7BT%7D%20%5Cmathbf%7Bx%7D_%7Bn%2C%3A%7D%20%5Cright%29%20%3D%20%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20%5Cleft%28%20%5Cleft%5B%5Cbegin%7Barray%7D%0A%7Brrrr%7D%0Ax_%7Bn%2C0%7D%5E%7B2%7D%20%26%20x_%7Bn%2C0%7Dx_%7Bn%2C1%7D%20%26%20%5Cldots%20%26%20x_%7Bn%2C0%7Dx_%7Bn%2CD%7D%20%5C%5C%0A%5Cvdots%20%26%20%5Cvdots%20%26%20%5Cddots%20%26%20%5Cvdots%5C%5C%0Ax_%7Bn%2CD%7Dx_%7Bn%2C0%7D%20%26%20x_%7Bn%2CD%7Dx_%7Bn%2C1%7D%20%26%20%5Cldots%20%26%20x_%7Bn%2CD%7D%5E2%20%0A%5Cend%7Barray%7D%5Cright%5D%20%5Cright%29%0A " 
\sum_{n=1}^{N} \left( \mathbf{x}_{n,:}^{T} \mathbf{x}_{n,:} \right) = \sum_{n=1}^{N} \left( \left[\begin{array}
{rrrr}
x_{n,0}^{2} & x_{n,0}x_{n,1} & \ldots & x_{n,0}x_{n,D} \\
\vdots & \vdots & \ddots & \vdots\\
x_{n,D}x_{n,0} & x_{n,D}x_{n,1} & \ldots & x_{n,D}^2 
\end{array}\right] \right)
")

The sum of squares matrix has ![D+1](https://latex.codecogs.com/png.latex?D%2B1 "D+1") rows and ![D+1](https://latex.codecogs.com/png.latex?D%2B1 "D+1") columns. It can be more compactly written as:

![ 
\\sum\_{n=1}^{N} \\left( \\mathbf{x}\_{n,:}^{T} \\mathbf{x}\_{n,:} \\right) = \\mathbf{X}^T \\mathbf{X}
](https://latex.codecogs.com/png.latex?%20%0A%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20%5Cleft%28%20%5Cmathbf%7Bx%7D_%7Bn%2C%3A%7D%5E%7BT%7D%20%5Cmathbf%7Bx%7D_%7Bn%2C%3A%7D%20%5Cright%29%20%3D%20%5Cmathbf%7BX%7D%5ET%20%5Cmathbf%7BX%7D%0A " 
\sum_{n=1}^{N} \left( \mathbf{x}_{n,:}^{T} \mathbf{x}_{n,:} \right) = \mathbf{X}^T \mathbf{X}
")

With the sum of squares matrix defined, the second summation term can be rewritten as:

![ 
\\mathbf{X}^{T} \\mathbf{X} \\boldsymbol{\\beta}
](https://latex.codecogs.com/png.latex?%20%0A%5Cmathbf%7BX%7D%5E%7BT%7D%20%5Cmathbf%7BX%7D%20%5Cboldsymbol%7B%5Cbeta%7D%0A " 
\mathbf{X}^{T} \mathbf{X} \boldsymbol{\beta}
")

The gradient vector of ![D+1](https://latex.codecogs.com/png.latex?D%2B1 "D+1") elements can therefore be written as:

![ 
\\mathbf{g} = \\frac{1}{\\sigma^2} \\left( \\mathbf{X}^{T} \\mathbf{y} - \\mathbf{X}^{T} \\mathbf{X} \\boldsymbol{\\beta} \\right)
](https://latex.codecogs.com/png.latex?%20%0A%5Cmathbf%7Bg%7D%20%3D%20%5Cfrac%7B1%7D%7B%5Csigma%5E2%7D%20%5Cleft%28%20%5Cmathbf%7BX%7D%5E%7BT%7D%20%5Cmathbf%7By%7D%20-%20%5Cmathbf%7BX%7D%5E%7BT%7D%20%5Cmathbf%7BX%7D%20%5Cboldsymbol%7B%5Cbeta%7D%20%5Cright%29%0A " 
\mathbf{g} = \frac{1}{\sigma^2} \left( \mathbf{X}^{T} \mathbf{y} - \mathbf{X}^{T} \mathbf{X} \boldsymbol{\beta} \right)
")

#### Posterior mode

The posterior mode corresponds to the ![\\hat{\\boldsymbol{\\beta}}](https://latex.codecogs.com/png.latex?%5Chat%7B%5Cboldsymbol%7B%5Cbeta%7D%7D "\hat{\boldsymbol{\beta}}") values where the gradient vector equal to zero, ![\\mathbf{g} = \\mathbf{0}](https://latex.codecogs.com/png.latex?%5Cmathbf%7Bg%7D%20%3D%20%5Cmathbf%7B0%7D "\mathbf{g} = \mathbf{0}"). Setting the above expression equal to zero and rearranging gives:

![ 
\\mathbf{X}^{T} \\mathbf{X} \\hat{\\boldsymbol{\\beta}} = \\mathbf{X}^{T} \\mathbf{y}
](https://latex.codecogs.com/png.latex?%20%0A%5Cmathbf%7BX%7D%5E%7BT%7D%20%5Cmathbf%7BX%7D%20%5Chat%7B%5Cboldsymbol%7B%5Cbeta%7D%7D%20%3D%20%5Cmathbf%7BX%7D%5E%7BT%7D%20%5Cmathbf%7By%7D%0A " 
\mathbf{X}^{T} \mathbf{X} \hat{\boldsymbol{\beta}} = \mathbf{X}^{T} \mathbf{y}
")

The posterior mode is therefore:

![ 
\\hat{\\boldsymbol{\\beta}} = \\left( \\mathbf{X}^{T} \\mathbf{X} \\right)^{-1} \\mathbf{X}^{T} \\mathbf{y}
](https://latex.codecogs.com/png.latex?%20%0A%5Chat%7B%5Cboldsymbol%7B%5Cbeta%7D%7D%20%3D%20%5Cleft%28%20%5Cmathbf%7BX%7D%5E%7BT%7D%20%5Cmathbf%7BX%7D%20%5Cright%29%5E%7B-1%7D%20%5Cmathbf%7BX%7D%5E%7BT%7D%20%5Cmathbf%7By%7D%0A " 
\hat{\boldsymbol{\beta}} = \left( \mathbf{X}^{T} \mathbf{X} \right)^{-1} \mathbf{X}^{T} \mathbf{y}
")

The expression for the posterior mode has several important features. First, the mode does not depend on the noise, ![\\sigma](https://latex.codecogs.com/png.latex?%5Csigma "\sigma"). Second, we derived this expression assuming an infinitely diffuse or vague prior. Based on what we saw last week, under this assumption the posterior mode converges to the maximum likelihood estimate (MLE). This should have been obvious here, since we explicitely set the log-prior to zero at the start of our derivation. Lastly, as pointed out at the beginning of the derivation, the log-likelihood can be written in terms of the sum of squared errors. Thus, our MLE is equivalent to the Ordinary Least Squares (OLS) estimate which minimizes the squared error loss between the observations and the linear predictor.

#### Hessian

To complete the Laplace approximation, we need to derive the Hessian, or the matrix of second derivatives. For our specific example, with ![\\beta\_{0}](https://latex.codecogs.com/png.latex?%5Cbeta_%7B0%7D "\beta_{0}") and ![\\beta\_{1}](https://latex.codecogs.com/png.latex?%5Cbeta_%7B1%7D "\beta_{1}"), the Hessian matrix is a ![2\\times 2](https://latex.codecogs.com/png.latex?2%5Ctimes%202 "2\times 2") matrix:

![ 
\\mathbf{H} = \\left\[\\begin{array}
{rr}
\\frac{\\partial^{2} L}{\\partial \\beta\_{0}^2} & \\frac{\\partial^{2}L}{\\partial \\beta\_{0} \\partial \\beta\_{1}} \\\\
\\frac{\\partial^{2}L}{\\partial \\beta\_{1} \\partial \\beta\_{0}} & \\frac{\\partial^{2} L}{\\partial \\beta\_{1}^2} 
\\end{array}\\right\]
](https://latex.codecogs.com/png.latex?%20%0A%5Cmathbf%7BH%7D%20%3D%20%5Cleft%5B%5Cbegin%7Barray%7D%0A%7Brr%7D%0A%5Cfrac%7B%5Cpartial%5E%7B2%7D%20L%7D%7B%5Cpartial%20%5Cbeta_%7B0%7D%5E2%7D%20%26%20%5Cfrac%7B%5Cpartial%5E%7B2%7DL%7D%7B%5Cpartial%20%5Cbeta_%7B0%7D%20%5Cpartial%20%5Cbeta_%7B1%7D%7D%20%5C%5C%0A%5Cfrac%7B%5Cpartial%5E%7B2%7DL%7D%7B%5Cpartial%20%5Cbeta_%7B1%7D%20%5Cpartial%20%5Cbeta_%7B0%7D%7D%20%26%20%5Cfrac%7B%5Cpartial%5E%7B2%7D%20L%7D%7B%5Cpartial%20%5Cbeta_%7B1%7D%5E2%7D%20%0A%5Cend%7Barray%7D%5Cright%5D%0A " 
\mathbf{H} = \left[\begin{array}
{rr}
\frac{\partial^{2} L}{\partial \beta_{0}^2} & \frac{\partial^{2}L}{\partial \beta_{0} \partial \beta_{1}} \\
\frac{\partial^{2}L}{\partial \beta_{1} \partial \beta_{0}} & \frac{\partial^{2} L}{\partial \beta_{1}^2} 
\end{array}\right]
")

We can determine the Hessian matrix by inspection of the gradient vector expression. Notice that the only portion impacted by ![\\boldsymbol{\\beta}](https://latex.codecogs.com/png.latex?%5Cboldsymbol%7B%5Cbeta%7D "\boldsymbol{\beta}") is the term involving the sum-of-squares matrix. Differentiating the gradient vector with respect to each element gives:

![ 
\\mathbf{H} = -\\frac{1}{\\sigma^2} \\mathbf{X}^{T} \\mathbf{X}
](https://latex.codecogs.com/png.latex?%20%0A%5Cmathbf%7BH%7D%20%3D%20-%5Cfrac%7B1%7D%7B%5Csigma%5E2%7D%20%5Cmathbf%7BX%7D%5E%7BT%7D%20%5Cmathbf%7BX%7D%0A " 
\mathbf{H} = -\frac{1}{\sigma^2} \mathbf{X}^{T} \mathbf{X}
")

The posterior covariance matrix on all ![D+1](https://latex.codecogs.com/png.latex?D%2B1 "D+1") parameters is then equal to the negative inverse of the Hessian matrix:

![ 
\\mathrm{cov}\\left( \\boldsymbol{\\beta},\\boldsymbol{\\beta} \\right) = \\sigma^2 \\left( \\mathbf{X}^{T} \\mathbf{X} \\right)^{-1}
](https://latex.codecogs.com/png.latex?%20%0A%5Cmathrm%7Bcov%7D%5Cleft%28%20%5Cboldsymbol%7B%5Cbeta%7D%2C%5Cboldsymbol%7B%5Cbeta%7D%20%5Cright%29%20%3D%20%5Csigma%5E2%20%5Cleft%28%20%5Cmathbf%7BX%7D%5E%7BT%7D%20%5Cmathbf%7BX%7D%20%5Cright%29%5E%7B-1%7D%0A " 
\mathrm{cov}\left( \boldsymbol{\beta},\boldsymbol{\beta} \right) = \sigma^2 \left( \mathbf{X}^{T} \mathbf{X} \right)^{-1}
")

There are several important points to make about the posterior covariance matrix. First, for a given noise term, ![\\sigma](https://latex.codecogs.com/png.latex?%5Csigma "\sigma"), the responses, ![\\mathbf{y}](https://latex.codecogs.com/png.latex?%5Cmathbf%7By%7D "\mathbf{y}"), do not influence the posterior uncertainty. The covariance structure between the ![D+1](https://latex.codecogs.com/png.latex?D%2B1 "D+1") parameters is impacted by the design matrix through the sum-of-squares matrix. Presumably, the way we collect the data impacts our posterior uncertainty, and that is exactly what the posterior covariance matrix expression tells us. We could imagine that we could decide *optimal* input configurations which give us the same level of posterior uncertainty as randomly collecting observations. This concept is at the heart of Experimental Design procedures. Understanding the mathematical structure of the sum-of-squares matrix is at the heart of the designs which satisfy achieve the various *alphabet-optimality* conditions.

Therefore, starting from an infinitely diffuse prior, the posterior distribution on the ![D+1](https://latex.codecogs.com/png.latex?D%2B1 "D+1") parameters is the following MVN distribution:

![ 
\\boldsymbol{\\beta} \\mid \\mathbf{X},\\mathbf{y},\\sigma \\sim \\mathcal{N} \\left( \\boldsymbol{\\beta} \\mid \\left( \\mathbf{X}^{T} \\mathbf{X} \\right)^{-1} \\mathbf{X}^{T} \\mathbf{y} , \\sigma^2 \\left( \\mathbf{X}^{T} \\mathbf{X} \\right)^{-1} \\right)
](https://latex.codecogs.com/png.latex?%20%0A%5Cboldsymbol%7B%5Cbeta%7D%20%5Cmid%20%5Cmathbf%7BX%7D%2C%5Cmathbf%7By%7D%2C%5Csigma%20%5Csim%20%5Cmathcal%7BN%7D%20%5Cleft%28%20%5Cboldsymbol%7B%5Cbeta%7D%20%5Cmid%20%5Cleft%28%20%5Cmathbf%7BX%7D%5E%7BT%7D%20%5Cmathbf%7BX%7D%20%5Cright%29%5E%7B-1%7D%20%5Cmathbf%7BX%7D%5E%7BT%7D%20%5Cmathbf%7By%7D%20%2C%20%5Csigma%5E2%20%5Cleft%28%20%5Cmathbf%7BX%7D%5E%7BT%7D%20%5Cmathbf%7BX%7D%20%5Cright%29%5E%7B-1%7D%20%5Cright%29%0A " 
\boldsymbol{\beta} \mid \mathbf{X},\mathbf{y},\sigma \sim \mathcal{N} \left( \boldsymbol{\beta} \mid \left( \mathbf{X}^{T} \mathbf{X} \right)^{-1} \mathbf{X}^{T} \mathbf{y} , \sigma^2 \left( \mathbf{X}^{T} \mathbf{X} \right)^{-1} \right)
")

### Example

Let's apply the formulas we just derived to our current synthetic data example. As we have done throughout the last few weeks, we wil define a function to perform the analysis. However, before creating that function, it is important to note that in `R` we **cannot** use `*` for matrix multiplication. `R` views the `*` operation as multiplying all elements together between two objects. In order to tell `R` we are interested in matrix multiplication, we must use the `%*%` operator. The code chunk below defines the function `laplace_diffuse()` below reads in a design matrix, `X`, a vector of responses, `y`, and the given noise, `sd_known`. The inverse of the sum-of-squares matrix is calculated and that inverse is used to calculate the posterior mode and posterior covariance matrix. Both quantities are returned as elements of a list, with names consistent with the `my_laplace()` function.

``` r
laplace_diffuse <- function(X, y, sd_known)
{
  iSS <- solve(t(X) %*% X)
  
  bhat <- iSS %*% t(X) %*% as.matrix(y)
  
  bcov <- sd_known^2 * iSS
  
  list(mode = as.vector(bhat),
       var_matrix = bcov)
}
```

We originally applied the grid approximation to the ![N=10](https://latex.codecogs.com/png.latex?N%3D10 "N=10") case, and so we will return to that situation now. We need to first assemble the design matrix, ![\\mathbf{X}](https://latex.codecogs.com/png.latex?%5Cmathbf%7BX%7D "\mathbf{X}"), by binding together a vector of ![\\mathbf{1}](https://latex.codecogs.com/png.latex?%5Cmathbf%7B1%7D "\mathbf{1}")'s for the intercept term with the values of the ![x](https://latex.codecogs.com/png.latex?x "x") input vector. In order to compare our result with the laplace approximation, we need to put together the observations with the more diffuse prior specification.

``` r
X_matrix <- cbind(rep(1, nrow(train_df)), train_df$x)

### check the class
class(X_matrix)
```

    ## [1] "matrix"

``` r
### set the information for the laplace approximation
info_diffuse_01 <- list(
  xobs = train_df$x,
  yobs = train_df$y,
  sigma = sigma_true,
  b0_mu = 0,
  b0_sd = 5,
  b1_mu = 0,
  b1_sd = 5
)
```

Perform the Laplace approximation from a starting guess of ![\\boldsymbol{\\beta} = \\{-1, -1\\}](https://latex.codecogs.com/png.latex?%5Cboldsymbol%7B%5Cbeta%7D%20%3D%20%5C%7B-1%2C%20-1%5C%7D "\boldsymbol{\beta} = \{-1, -1\}"):

``` r
laplace_approx_N10 <- my_laplace(c(-1, -1), lm_logpost_01, info_diffuse_01)
```

And now evaluate the formulas in the `laplace_diffuse()` function:

``` r
analytic_result_N10 <- laplace_diffuse(X_matrix, train_df$y, sigma_true)
```

Let's compare the identified posterior modes between the two approaches:

``` r
laplace_approx_N10$mode
```

    ## [1] -0.241377  1.162628

``` r
analytic_result_N10$mode
```

    ## [1] -0.2415318  1.1638857

And now let's compare the posterior covariance matrices:

``` r
laplace_approx_N10$var_matrix
```

    ##            [,1]       [,2]
    ## [1,] 0.02510488 0.00188907
    ## [2,] 0.00188907 0.02748110

``` r
analytic_result_N10$var_matrix
```

    ##            [,1]       [,2]
    ## [1,] 0.02513026 0.00189305
    ## [2,] 0.00189305 0.02751149

The Laplace approximation and the analytic formulas yielded very similar results. The values are not exact matches, but they are very close!

### Informative prior

As long as our informative prior is a MVN distribution, the posterior distribution ![\\boldsymbol{\\beta}](https://latex.codecogs.com/png.latex?%5Cboldsymbol%7B%5Cbeta%7D "\boldsymbol{\beta}") will be a MVN. We will not go through the derivation, but it is important to note that the formulation is consistent with fitting the MVN distribution with a known covariance matrix. Let's define our informative prior MVN with prior mean vector ![\\mathbf{b}\_0](https://latex.codecogs.com/png.latex?%5Cmathbf%7Bb%7D_0 "\mathbf{b}_0") and prior covariance matrix ![\\mathbf{B}\_0](https://latex.codecogs.com/png.latex?%5Cmathbf%7BB%7D_0 "\mathbf{B}_0"):

![ 
\\boldsymbol{\\beta} \\mid \\mathbf{b}\_0, \\mathbf{B}\_0 \\sim \\mathcal{N} \\left( \\boldsymbol{\\beta} \\mid \\mathbf{b}\_0, \\mathbf{B}\_0 \\right)
](https://latex.codecogs.com/png.latex?%20%0A%5Cboldsymbol%7B%5Cbeta%7D%20%5Cmid%20%5Cmathbf%7Bb%7D_0%2C%20%5Cmathbf%7BB%7D_0%20%5Csim%20%5Cmathcal%7BN%7D%20%5Cleft%28%20%5Cboldsymbol%7B%5Cbeta%7D%20%5Cmid%20%5Cmathbf%7Bb%7D_0%2C%20%5Cmathbf%7BB%7D_0%20%5Cright%29%0A " 
\boldsymbol{\beta} \mid \mathbf{b}_0, \mathbf{B}_0 \sim \mathcal{N} \left( \boldsymbol{\beta} \mid \mathbf{b}_0, \mathbf{B}_0 \right)
")

After observing ![N](https://latex.codecogs.com/png.latex?N "N") observations with response vector ![\\mathbf{y}](https://latex.codecogs.com/png.latex?%5Cmathbf%7By%7D "\mathbf{y}") and design matrix ![\\mathbf{X}](https://latex.codecogs.com/png.latex?%5Cmathbf%7BX%7D "\mathbf{X}") the posterior MVN distribution on the unknown parameters ![\\boldsymbol{\\beta}](https://latex.codecogs.com/png.latex?%5Cboldsymbol%7B%5Cbeta%7D "\boldsymbol{\beta}") is:

![ 
\\boldsymbol{\\beta} \\mid \\mathbf{X},\\mathbf{y},\\sigma \\sim \\mathcal{N} \\left( \\boldsymbol{\\beta} \\mid \\mathbf{b}\_N, \\mathbf{B}\_N \\right)
](https://latex.codecogs.com/png.latex?%20%0A%5Cboldsymbol%7B%5Cbeta%7D%20%5Cmid%20%5Cmathbf%7BX%7D%2C%5Cmathbf%7By%7D%2C%5Csigma%20%5Csim%20%5Cmathcal%7BN%7D%20%5Cleft%28%20%5Cboldsymbol%7B%5Cbeta%7D%20%5Cmid%20%5Cmathbf%7Bb%7D_N%2C%20%5Cmathbf%7BB%7D_N%20%5Cright%29%0A " 
\boldsymbol{\beta} \mid \mathbf{X},\mathbf{y},\sigma \sim \mathcal{N} \left( \boldsymbol{\beta} \mid \mathbf{b}_N, \mathbf{B}_N \right)
")

The posterior precision matrix is the sum of the prior precision and data precision matrices. But, note that the now the data precision depends on the design matrix!

![ 
\\mathbf{B}\_{N}^{-1} = \\mathbf{B}\_{0}^{-1} + \\frac{1}{\\sigma^2} \\mathbf{X}^{T} \\mathbf{X}
](https://latex.codecogs.com/png.latex?%20%0A%5Cmathbf%7BB%7D_%7BN%7D%5E%7B-1%7D%20%3D%20%5Cmathbf%7BB%7D_%7B0%7D%5E%7B-1%7D%20%2B%20%5Cfrac%7B1%7D%7B%5Csigma%5E2%7D%20%5Cmathbf%7BX%7D%5E%7BT%7D%20%5Cmathbf%7BX%7D%0A " 
\mathbf{B}_{N}^{-1} = \mathbf{B}_{0}^{-1} + \frac{1}{\sigma^2} \mathbf{X}^{T} \mathbf{X}
")

The posterior mean is a precision weighted average of the prior mean and the observations, just as we have seen before. The posterior mean formula is:

![ 
\\mathbf{b}\_N = \\mathbf{B}\_N \\left( \\mathbf{B}\_{0}^{-1} \\mathbf{b}\_{0} + \\frac{1}{\\sigma^2} \\mathbf{X}^{T} \\mathbf{y} \\right)
](https://latex.codecogs.com/png.latex?%20%0A%5Cmathbf%7Bb%7D_N%20%3D%20%5Cmathbf%7BB%7D_N%20%5Cleft%28%20%5Cmathbf%7BB%7D_%7B0%7D%5E%7B-1%7D%20%5Cmathbf%7Bb%7D_%7B0%7D%20%2B%20%5Cfrac%7B1%7D%7B%5Csigma%5E2%7D%20%5Cmathbf%7BX%7D%5E%7BT%7D%20%5Cmathbf%7By%7D%20%5Cright%29%0A " 
\mathbf{b}_N = \mathbf{B}_N \left( \mathbf{B}_{0}^{-1} \mathbf{b}_{0} + \frac{1}{\sigma^2} \mathbf{X}^{T} \mathbf{y} \right)
")

**Can you see how the MLE, ![\\hat{\\boldsymbol{\\beta}}](https://latex.codecogs.com/png.latex?%5Chat%7B%5Cboldsymbol%7B%5Cbeta%7D%7D "\hat{\boldsymbol{\beta}}"), is recovered when when we use an infinitely vague prior?**

Unknown ![\\sigma](https://latex.codecogs.com/png.latex?%5Csigma "\sigma")
--------------------------------------------------------------------------

All of the previous analysis and discussion assumed that we knew ![\\sigma](https://latex.codecogs.com/png.latex?%5Csigma "\sigma"). In a real problem though, we do not anticipate knowing the noise. It served as a useful first step to help us understand key concepts such as the importance of the design matrix, ![\\mathbf{X}](https://latex.codecogs.com/png.latex?%5Cmathbf%7BX%7D "\mathbf{X}"), on the posterior solution. But how can we build a model when the noise is unknown?

### MLE

We will first consider the MLE on ![\\sigma](https://latex.codecogs.com/png.latex?%5Csigma "\sigma"), or rather on the variance, ![\\sigma^2](https://latex.codecogs.com/png.latex?%5Csigma%5E2 "\sigma^2"). Can you make a guess for what an estimate to ![\\sigma^2](https://latex.codecogs.com/png.latex?%5Csigma%5E2 "\sigma^2") could be? We will quickly derive the MLE on ![\\sigma^2](https://latex.codecogs.com/png.latex?%5Csigma%5E2 "\sigma^2") below. Start by writing out the log-likelihood again, but this time including all terms involving either ![\\sigma^2](https://latex.codecogs.com/png.latex?%5Csigma%5E2 "\sigma^2") and or ![\\boldsymbol{\\beta}](https://latex.codecogs.com/png.latex?%5Cboldsymbol%7B%5Cbeta%7D "\boldsymbol{\beta}"):

![ 
L \\propto -\\frac{N}{2} \\log\\left\[\\sigma^2\\right\] - \\frac{1}{2\\sigma^2} \\left( \\mathbf{y} - \\mathbf{X}\\boldsymbol{\\beta} \\right)^T \\left( \\mathbf{y} - \\mathbf{X}\\boldsymbol{\\beta} \\right)
](https://latex.codecogs.com/png.latex?%20%0AL%20%5Cpropto%20-%5Cfrac%7BN%7D%7B2%7D%20%5Clog%5Cleft%5B%5Csigma%5E2%5Cright%5D%20-%20%5Cfrac%7B1%7D%7B2%5Csigma%5E2%7D%20%5Cleft%28%20%5Cmathbf%7By%7D%20-%20%5Cmathbf%7BX%7D%5Cboldsymbol%7B%5Cbeta%7D%20%5Cright%29%5ET%20%5Cleft%28%20%5Cmathbf%7By%7D%20-%20%5Cmathbf%7BX%7D%5Cboldsymbol%7B%5Cbeta%7D%20%5Cright%29%0A " 
L \propto -\frac{N}{2} \log\left[\sigma^2\right] - \frac{1}{2\sigma^2} \left( \mathbf{y} - \mathbf{X}\boldsymbol{\beta} \right)^T \left( \mathbf{y} - \mathbf{X}\boldsymbol{\beta} \right)
")

Evaluate the log-likelihood at the MLE on the linear predictor parameters:

![ 
L \\rvert\_{\\hat{\\boldsymbol{\\beta}}} \\propto -\\frac{N}{2} \\log\\left\[\\sigma^2\\right\] - \\frac{1}{2\\sigma^2} \\left( \\mathbf{y} - \\mathbf{X}\\hat{\\boldsymbol{\\beta}} \\right)^T \\left( \\mathbf{y} - \\mathbf{X}\\hat{\\boldsymbol{\\beta}} \\right)
](https://latex.codecogs.com/png.latex?%20%0AL%20%5Crvert_%7B%5Chat%7B%5Cboldsymbol%7B%5Cbeta%7D%7D%7D%20%5Cpropto%20-%5Cfrac%7BN%7D%7B2%7D%20%5Clog%5Cleft%5B%5Csigma%5E2%5Cright%5D%20-%20%5Cfrac%7B1%7D%7B2%5Csigma%5E2%7D%20%5Cleft%28%20%5Cmathbf%7By%7D%20-%20%5Cmathbf%7BX%7D%5Chat%7B%5Cboldsymbol%7B%5Cbeta%7D%7D%20%5Cright%29%5ET%20%5Cleft%28%20%5Cmathbf%7By%7D%20-%20%5Cmathbf%7BX%7D%5Chat%7B%5Cboldsymbol%7B%5Cbeta%7D%7D%20%5Cright%29%0A " 
L \rvert_{\hat{\boldsymbol{\beta}}} \propto -\frac{N}{2} \log\left[\sigma^2\right] - \frac{1}{2\sigma^2} \left( \mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}} \right)^T \left( \mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}} \right)
")

 Take the derivative of the log-likelihood evaluated at ![\\hat{\\boldsymbol{\\beta}}](https://latex.codecogs.com/png.latex?%5Chat%7B%5Cboldsymbol%7B%5Cbeta%7D%7D "\hat{\boldsymbol{\beta}}") with respect to ![\\sigma^2](https://latex.codecogs.com/png.latex?%5Csigma%5E2 "\sigma^2"):

![ 
\\frac{\\partial L\\rvert\_{\\hat{\\boldsymbol{\\beta}}}}{\\partial \\left( \\sigma^2\\right)} \\propto -\\frac{N}{2} \\frac{1}{\\sigma^2} -\\frac{1}{2} \\left(-\\frac{1}{\\left( \\sigma^2 \\right)^2}\\right) \\left( \\mathbf{y} - \\mathbf{X}\\hat{\\boldsymbol{\\beta}} \\right)^T \\left( \\mathbf{y} - \\mathbf{X}\\hat{\\boldsymbol{\\beta}} \\right)
](https://latex.codecogs.com/png.latex?%20%0A%5Cfrac%7B%5Cpartial%20L%5Crvert_%7B%5Chat%7B%5Cboldsymbol%7B%5Cbeta%7D%7D%7D%7D%7B%5Cpartial%20%5Cleft%28%20%5Csigma%5E2%5Cright%29%7D%20%5Cpropto%20-%5Cfrac%7BN%7D%7B2%7D%20%5Cfrac%7B1%7D%7B%5Csigma%5E2%7D%20-%5Cfrac%7B1%7D%7B2%7D%20%5Cleft%28-%5Cfrac%7B1%7D%7B%5Cleft%28%20%5Csigma%5E2%20%5Cright%29%5E2%7D%5Cright%29%20%5Cleft%28%20%5Cmathbf%7By%7D%20-%20%5Cmathbf%7BX%7D%5Chat%7B%5Cboldsymbol%7B%5Cbeta%7D%7D%20%5Cright%29%5ET%20%5Cleft%28%20%5Cmathbf%7By%7D%20-%20%5Cmathbf%7BX%7D%5Chat%7B%5Cboldsymbol%7B%5Cbeta%7D%7D%20%5Cright%29%0A " 
\frac{\partial L\rvert_{\hat{\boldsymbol{\beta}}}}{\partial \left( \sigma^2\right)} \propto -\frac{N}{2} \frac{1}{\sigma^2} -\frac{1}{2} \left(-\frac{1}{\left( \sigma^2 \right)^2}\right) \left( \mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}} \right)^T \left( \mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}} \right)
")

Factor the right hand side into two terms:

![ 
\\frac{\\partial L\\rvert\_{\\hat{\\boldsymbol{\\beta}}}}{\\partial \\left( \\sigma^2\\right)} \\propto \\left( -\\frac{1}{2\\sigma^2} \\right) \\left(N - \\frac{1}{\\sigma^2} \\left( \\mathbf{y} - \\mathbf{X}\\hat{\\boldsymbol{\\beta}} \\right)^T \\left( \\mathbf{y} - \\mathbf{X}\\hat{\\boldsymbol{\\beta}} \\right) \\right)
](https://latex.codecogs.com/png.latex?%20%0A%5Cfrac%7B%5Cpartial%20L%5Crvert_%7B%5Chat%7B%5Cboldsymbol%7B%5Cbeta%7D%7D%7D%7D%7B%5Cpartial%20%5Cleft%28%20%5Csigma%5E2%5Cright%29%7D%20%5Cpropto%20%5Cleft%28%20-%5Cfrac%7B1%7D%7B2%5Csigma%5E2%7D%20%5Cright%29%20%5Cleft%28N%20-%20%5Cfrac%7B1%7D%7B%5Csigma%5E2%7D%20%5Cleft%28%20%5Cmathbf%7By%7D%20-%20%5Cmathbf%7BX%7D%5Chat%7B%5Cboldsymbol%7B%5Cbeta%7D%7D%20%5Cright%29%5ET%20%5Cleft%28%20%5Cmathbf%7By%7D%20-%20%5Cmathbf%7BX%7D%5Chat%7B%5Cboldsymbol%7B%5Cbeta%7D%7D%20%5Cright%29%20%5Cright%29%0A " 
\frac{\partial L\rvert_{\hat{\boldsymbol{\beta}}}}{\partial \left( \sigma^2\right)} \propto \left( -\frac{1}{2\sigma^2} \right) \left(N - \frac{1}{\sigma^2} \left( \mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}} \right)^T \left( \mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}} \right) \right)
")

 The derivative of the log-likelihood with respect to ![\\sigma^2](https://latex.codecogs.com/png.latex?%5Csigma%5E2 "\sigma^2") can therefore be zero either at infinite variance, or if the expression within the second set of parantheses on the right hand side is equal to zero.

![ 
0=N - \\frac{1}{\\hat{\\sigma}^2} \\left( \\mathbf{y} - \\mathbf{X}\\hat{\\boldsymbol{\\beta}} \\right)^T \\left( \\mathbf{y} - \\mathbf{X}\\hat{\\boldsymbol{\\beta}} \\right)
](https://latex.codecogs.com/png.latex?%20%0A0%3DN%20-%20%5Cfrac%7B1%7D%7B%5Chat%7B%5Csigma%7D%5E2%7D%20%5Cleft%28%20%5Cmathbf%7By%7D%20-%20%5Cmathbf%7BX%7D%5Chat%7B%5Cboldsymbol%7B%5Cbeta%7D%7D%20%5Cright%29%5ET%20%5Cleft%28%20%5Cmathbf%7By%7D%20-%20%5Cmathbf%7BX%7D%5Chat%7B%5Cboldsymbol%7B%5Cbeta%7D%7D%20%5Cright%29%0A " 
0=N - \frac{1}{\hat{\sigma}^2} \left( \mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}} \right)^T \left( \mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}} \right)
")

 The above expression can be easily solved for the MLE on ![\\sigma^2](https://latex.codecogs.com/png.latex?%5Csigma%5E2 "\sigma^2"):

![
\\hat{\\sigma}^2 = \\frac{1}{N} \\left( \\mathbf{y} - \\mathbf{X}\\hat{\\boldsymbol{\\beta}} \\right)^T \\left( \\mathbf{y} - \\mathbf{X}\\hat{\\boldsymbol{\\beta}} \\right)
](https://latex.codecogs.com/png.latex?%0A%5Chat%7B%5Csigma%7D%5E2%20%3D%20%5Cfrac%7B1%7D%7BN%7D%20%5Cleft%28%20%5Cmathbf%7By%7D%20-%20%5Cmathbf%7BX%7D%5Chat%7B%5Cboldsymbol%7B%5Cbeta%7D%7D%20%5Cright%29%5ET%20%5Cleft%28%20%5Cmathbf%7By%7D%20-%20%5Cmathbf%7BX%7D%5Chat%7B%5Cboldsymbol%7B%5Cbeta%7D%7D%20%5Cright%29%0A "
\hat{\sigma}^2 = \frac{1}{N} \left( \mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}} \right)^T \left( \mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}} \right)
")

 Although written in matrix notation, the above expression is simply saying that the MLE on ![\\sigma^2](https://latex.codecogs.com/png.latex?%5Csigma%5E2 "\sigma^2") is the mean squared error of the linear predictor MLE! **Can you see why?**

The impact of having to estimate ![\\sigma^2](https://latex.codecogs.com/png.latex?%5Csigma%5E2 "\sigma^2") compared to knowing it exactly, within the MLE framework, is that the uncertainty intervals on the linear predictor parameters ![\\boldsymbol{\\beta}](https://latex.codecogs.com/png.latex?%5Cboldsymbol%7B%5Cbeta%7D "\boldsymbol{\beta}") are widened.

### Bayesian formulation

In a Bayesian formulation, we must specify a prior on the unknown ![\\sigma](https://latex.codecogs.com/png.latex?%5Csigma "\sigma") term, in addition to our prior on the linear predictor parameters, ![\\boldsymbol{\\beta}](https://latex.codecogs.com/png.latex?%5Cboldsymbol%7B%5Cbeta%7D "\boldsymbol{\beta}"). A benefit of the Bayesian approach over the maximum likelihood estimates is that we will have a probabilistic statement about the noise term, rather than just an estimate.

In terms of our running example, having an unknown ![\\sigma](https://latex.codecogs.com/png.latex?%5Csigma "\sigma") means that we now have 3 unknown parameters to learn. If we would continue to use the Grid approximation, and maintain 251 unique values in all 3 dimensions we would have to evaluate the log-posterior 1.5810^{7} times! Although we could reduce the resolution of our grid to reduce the required number of evaluations, a method like the Laplace approximation allows us to eventually scale up from a single input to multiple inputs within our linear model.

However, as we discussed last week, we cannot simply apply the Laplace approximation to the noise term. The approximate MVN distribution will not respect the natural lower limit on the standard deviation. We will need to apply a transformation and properly account for the change-of-variables within our posterior. Last week we applied the logit-transformation to respect both a lower and upper bound. This week, we will use a log-transformation, which does not impose an upper bound constraint.

#### Probability model

With the transformation, we will now learn the linear predictor parameters, ![\\boldsymbol{\\beta}](https://latex.codecogs.com/png.latex?%5Cboldsymbol%7B%5Cbeta%7D "\boldsymbol{\beta}"), and the unbounded transformed standard deviation, ![\\phi = \\log\\left\[\\sigma\\right\]](https://latex.codecogs.com/png.latex?%5Cphi%20%3D%20%5Clog%5Cleft%5B%5Csigma%5Cright%5D "\phi = \log\left[\sigma\right]"). We will specify our prior belief on ![\\sigma](https://latex.codecogs.com/png.latex?%5Csigma "\sigma") directly, and assume it is uniformly distributed between 0 and 10. Although the log-transformation does not "see" an upper bounded constraint, the uniform max value of 10 is quite far from our known *true* ![\\sigma](https://latex.codecogs.com/png.latex?%5Csigma "\sigma") value which generated the synthetic observations. In a real problem, we will obviously not have the luxury of knowing how far away our upper bound is from the *truth*. We will discuss *scaling* issues in a later lecture.

To complete our probability model, we will continue use to independent normal distributions on the ![\\beta\_{0}](https://latex.codecogs.com/png.latex?%5Cbeta_%7B0%7D "\beta_{0}") and ![\\beta\_{1}](https://latex.codecogs.com/png.latex?%5Cbeta_%7B1%7D "\beta_{1}"). We will use a compromise between the informative and the diffuse priors discussed earlier, and so set the prior standard deviation to 2.5.

The complete probability model is therefore:

![ 
y\_{n} \\mid \\mu\_{n}, \\phi \\sim \\mathrm{normal}\\left(y\_{n} \\mid \\mu\_{n}, \\exp\\left(\\phi\\right) \\right) \\\\ \\mu\_{n} = \\beta\_{0} + \\beta\_{1}x\_{n} \\\\ \\beta\_{0} \\sim \\mathrm{normal}\\left( \\beta\_{0} \\mid 0, 2.5 \\right) \\\\ \\beta\_{1} \\sim \\mathrm{normal}\\left( \\beta\_{1} \\mid 0, 2.5 \\right) \\\\ \\exp\\left( \\phi \\right) \\sim \\mathrm{uniform}\\left(\\exp\\left( \\phi \\right), 0, 10 \\right)
](https://latex.codecogs.com/png.latex?%20%0Ay_%7Bn%7D%20%5Cmid%20%5Cmu_%7Bn%7D%2C%20%5Cphi%20%5Csim%20%5Cmathrm%7Bnormal%7D%5Cleft%28y_%7Bn%7D%20%5Cmid%20%5Cmu_%7Bn%7D%2C%20%5Cexp%5Cleft%28%5Cphi%5Cright%29%20%5Cright%29%20%5C%5C%20%5Cmu_%7Bn%7D%20%3D%20%5Cbeta_%7B0%7D%20%2B%20%5Cbeta_%7B1%7Dx_%7Bn%7D%20%5C%5C%20%5Cbeta_%7B0%7D%20%5Csim%20%5Cmathrm%7Bnormal%7D%5Cleft%28%20%5Cbeta_%7B0%7D%20%5Cmid%200%2C%202.5%20%5Cright%29%20%5C%5C%20%5Cbeta_%7B1%7D%20%5Csim%20%5Cmathrm%7Bnormal%7D%5Cleft%28%20%5Cbeta_%7B1%7D%20%5Cmid%200%2C%202.5%20%5Cright%29%20%5C%5C%20%5Cexp%5Cleft%28%20%5Cphi%20%5Cright%29%20%5Csim%20%5Cmathrm%7Buniform%7D%5Cleft%28%5Cexp%5Cleft%28%20%5Cphi%20%5Cright%29%2C%200%2C%2010%20%5Cright%29%0A " 
y_{n} \mid \mu_{n}, \phi \sim \mathrm{normal}\left(y_{n} \mid \mu_{n}, \exp\left(\phi\right) \right) \\ \mu_{n} = \beta_{0} + \beta_{1}x_{n} \\ \beta_{0} \sim \mathrm{normal}\left( \beta_{0} \mid 0, 2.5 \right) \\ \beta_{1} \sim \mathrm{normal}\left( \beta_{1} \mid 0, 2.5 \right) \\ \exp\left( \phi \right) \sim \mathrm{uniform}\left(\exp\left( \phi \right), 0, 10 \right)
")

#### Laplace approximation

We can now encode our probability model within a function which evaluates the log-posterior for a given set of parameter values. As we used last week, to denote that a change-of-variables transformation was applied, the first argument to the `lm_logpost_02()` function is a vector named `phi`. The parameters are ordered such that the first two elements of `phi` are ![\\beta\_{0}](https://latex.codecogs.com/png.latex?%5Cbeta_%7B0%7D "\beta_{0}") and ![beta\_{1}](https://latex.codecogs.com/png.latex?beta_%7B1%7D "beta_{1}"), and the last element is the unbounded standard deviation ![\\phi](https://latex.codecogs.com/png.latex?%5Cphi "\phi"). The second argument, `my_info`, is again a list which contains all necessary information for evaluating the log-posterior function. After evaluating the log-likelihood and log-priors, the transformation is accounted for by adding in the log of the derivative of the inverse transformation function.

``` r
lm_logpost_02 <- function(phi, my_info)
{
  # unpack the parameter vector
  beta_0 <- phi[1]
  beta_1 <- phi[2]
  # back-transform from phi to sigma
  lik_sigma <- exp(phi[3])
  
  # calculate linear predictor
  mu <- beta_0 + beta_1 * my_info$xobs
  
  # evaluate the log-likelihood
  log_lik <- sum(dnorm(x = my_info$yobs,
                       mean = mu,
                       sd = lik_sigma,
                       log = TRUE))
  
  # evaluate the log-prior
  log_prior <- dnorm(x = beta_0,
                     mean = my_info$b0_mu,
                     sd = my_info$b0_sd,
                     log = TRUE) +
    dnorm(x = beta_1,
          mean = my_info$b1_mu,
          sd = my_info$b1_sd,
          log = TRUE) +
    dunif(x = lik_sigma,
          min = my_info$sigma_lwr,
          max = my_info$sigma_upr,
          log = TRUE)
  
  # account for the transformation
  log_derive_adjust <- phi[3]
  
  # sum together
  log_lik + log_prior + log_derive_adjust
}
```

Our list of required information is specified in the code chunk below:

``` r
info_use_02 <- list(
  xobs = train_df$x,
  yobs = train_df$y,
  sigma_lwr = 0,
  sigma_upr = 10,
  b0_mu = 0,
  b0_sd = 2.5,
  b1_mu = 0,
  b1_sd = 2.5
)
```

We will try out two different initial guesses, to check the robustness of our posterior mode. For the first guess use ![\\{-1, -1, \\log\\left\[2\\right\] \\}](https://latex.codecogs.com/png.latex?%5C%7B-1%2C%20-1%2C%20%5Clog%5Cleft%5B2%5Cright%5D%20%5C%7D "\{-1, -1, \log\left[2\right] \}"), and for the second guess use ![\\{2.5, 2.5, \\log\\left\[1\\right\] \\}](https://latex.codecogs.com/png.latex?%5C%7B2.5%2C%202.5%2C%20%5Clog%5Cleft%5B1%5Cright%5D%20%5C%7D "\{2.5, 2.5, \log\left[1\right] \}"). Execute both laplace approximation attempts in the code chunk below.

``` r
laplace_unknown_sigma_aa <- my_laplace(c(-1, -1, log(2)), lm_logpost_02, info_use_02)

laplace_unknown_sigma_bb <- my_laplace(c(2.5, 2.5, log(1)), lm_logpost_02, info_use_02)
```

Let's first check if both attempts converged:

``` r
laplace_unknown_sigma_aa$converge
```

    ## [1] "YES"

``` r
laplace_unknown_sigma_bb$converge
```

    ## [1] "YES"

The posterior modes are identical, as shown in the output below:

``` r
laplace_unknown_sigma_aa$mode
```

    ## [1] -0.2409766  1.1593660 -0.7466703

``` r
laplace_unknown_sigma_bb$mode
```

    ## [1] -0.2409757  1.1593660 -0.7466725

The (approximate) posterior covariance matrices for the two different initial guesses are also very similar. It appears that the solution is robust!

``` r
laplace_unknown_sigma_aa$var_matrix
```

    ##              [,1]          [,2]          [,3]
    ## [1,] 2.249747e-02  0.0016875221  6.160708e-05
    ## [2,] 1.687522e-03  0.0246253046 -5.003156e-04
    ## [3,] 6.160708e-05 -0.0005003156  5.556611e-02

``` r
laplace_unknown_sigma_bb$var_matrix
```

    ##              [,1]          [,2]          [,3]
    ## [1,] 2.249738e-02  0.0016875139  6.171669e-05
    ## [2,] 1.687514e-03  0.0246251989 -5.003161e-04
    ## [3,] 6.171669e-05 -0.0005003161  5.556587e-02

If we would be interested in printing to screen the posterior standard deviation associated with each parameter, we can use the `diag()` function to extract the elements along the main diagonal:

``` r
sqrt(diag(laplace_unknown_sigma_aa$var_matrix))
```

    ## [1] 0.1499916 0.1569245 0.2357246

As we discussed last week, this MVN distribution is the approximate posterior on the *unbounded* parameters. We will now generate random samples from this MVN and then back-transform the ![\\phi](https://latex.codecogs.com/png.latex?%5Cphi "\phi") samples to ![\\sigma](https://latex.codecogs.com/png.latex?%5Csigma "\sigma"), thereby properly accounting for the natural lower bound constrain on ![\\sigma](https://latex.codecogs.com/png.latex?%5Csigma "\sigma").

``` r
set.seed(4111)

post_unbound_samples <- MASS::mvrnorm(n = 1e4, 
                                      mu = laplace_unknown_sigma_aa$mode, 
                                      Sigma = laplace_unknown_sigma_aa$var_matrix) %>% 
  as.data.frame() %>% tbl_df() %>% 
  purrr::set_names(c("beta_0", "beta_1", "phi"))
```

All we have to do to back-transfrom ![\\phi](https://latex.codecogs.com/png.latex?%5Cphi "\phi") to ![\\sigma](https://latex.codecogs.com/png.latex?%5Csigma "\sigma") is to now apply the inverse-transformation function.

``` r
post_param_samples <- post_unbound_samples %>% 
  mutate(sigma = exp(phi))
```

Let's compare the marginal posterior distributions on each parameter with the known true values used to generate the observational data. The code chunk below first reshapes parameters to make it easy to specify the separate facet per parameter, before displaying the marginal posterior distribution with a histogram. The *true* values associated with each parameter are shown as dashed red vertical lines. As shown below, the posterior modes are all quite accurate relative to the *true* parameter values for this simple example!

``` r
post_param_samples %>% 
  select(-phi) %>% 
  tibble::rowid_to_column("post_id") %>% 
  tidyr::gather(key = "param_name", value = "value", -post_id) %>% 
  ggplot(mapping = aes(x = value)) +
  geom_histogram(mapping = aes(group = param_name),
                 bins = 55) +
  geom_vline(data = data.frame(true_value = c(beta_0_true, beta_1_true, sigma_true),
                               param_name = c("beta_0", "beta_1", "sigma")),
             mapping = aes(xintercept = true_value),
             color = "red", linetype = "dashed", size = 1.15) +
  facet_wrap(~param_name, scales = "free") +
  theme_bw() +
  theme(axis.text.y = element_blank())
```

![](lecture_07_github_files/figure-markdown_github/show_post_hist_compare_true-1.png)

Posterior predictions
---------------------

We will now evaluate the posterior predictive *distribution*. Rather than writing out integrals, let's just think through our modeling framework. We started out by defining parameter values and a grid of input "locations". We then evaluated the linear predictor, and then passed the linear predictor into a normal random number generator. We can follow the exact same workflow here by replacing "defining parameter values" with "select posterior sample". So to generate posterior predictions we will, first define a set of prediction (test) input values, ![\\mathbf{x}\_{\*} = \\{x\_{\*,1},...,x\_{\*,M} \\}](https://latex.codecogs.com/png.latex?%5Cmathbf%7Bx%7D_%7B%2A%7D%20%3D%20%5C%7Bx_%7B%2A%2C1%7D%2C...%2Cx_%7B%2A%2CM%7D%20%5C%7D "\mathbf{x}_{*} = \{x_{*,1},...,x_{*,M} \}"). For ![s = 1,...,S](https://latex.codecogs.com/png.latex?s%20%3D%201%2C...%2CS "s = 1,...,S"), select the ![s](https://latex.codecogs.com/png.latex?s "s")-th posterior parameter sample, ![\\{\\boldsymbol{\\beta}, \\sigma\\}\_s](https://latex.codecogs.com/png.latex?%5C%7B%5Cboldsymbol%7B%5Cbeta%7D%2C%20%5Csigma%5C%7D_s "\{\boldsymbol{\beta}, \sigma\}_s"). Calculate the linear predictor for the ![s](https://latex.codecogs.com/png.latex?s "s")-th posterior sample at each test input value, ![\\mathbf{\\mu}\_{\*s} = \\{ \\mu\_{\*s,1},...,\\mu\_{\*s,M} \\}](https://latex.codecogs.com/png.latex?%5Cmathbf%7B%5Cmu%7D_%7B%2As%7D%20%3D%20%5C%7B%20%5Cmu_%7B%2As%2C1%7D%2C...%2C%5Cmu_%7B%2As%2CM%7D%20%5C%7D "\mathbf{\mu}_{*s} = \{ \mu_{*s,1},...,\mu_{*s,M} \}"). Pass the linear predictor at the ![m](https://latex.codecogs.com/png.latex?m "m")-th input prediction "location" for the ![s](https://latex.codecogs.com/png.latex?s "s")-th posterior sample into a random number generater along with the ![s](https://latex.codecogs.com/png.latex?s "s")-th sample on ![\\sigma\_s](https://latex.codecogs.com/png.latex?%5Csigma_s "\sigma_s") to generate a random prediction, ![y\_{\*s,m}](https://latex.codecogs.com/png.latex?y_%7B%2As%2Cm%7D "y_{*s,m}").

This might sound convoluted, so let's break it down into the basic pieces. We first predict the linear predictor, in our case, the straight line ![\\mu](https://latex.codecogs.com/png.latex?%5Cmu "\mu"). Given ![\\mu](https://latex.codecogs.com/png.latex?%5Cmu "\mu"), we generate an observation ![y](https://latex.codecogs.com/png.latex?y "y"). We are therefore dealing with two distinct, yet equally important sources of uncertainty. To use its formal name, the first source is the *epistemic* source of uncertainty: the unknown parameter values. We do not know the *true* or *exact* parameter values. We only have statements of belief about their plausible values. So we try out many different plausible values. The second source is the *aleatory* uncertainty due to random chance. Although the *aleatory* uncertainty might seem the most natural, it is typically not addressed in regression problems.

There are several different ways we could code up the above steps. In either case, we must first define a set of prediction or test points. The code chunk below creates a vector, ![x\_test](https://latex.codecogs.com/png.latex?x_test "x_test"), of 25 points between the bounds of the original ![x](https://latex.codecogs.com/png.latex?x "x") vector.

``` r
x_test <- seq(range(x)[1], range(x)[2], length.out = 25)
```

We will now define a function which calculates the linear predictor and generates a random observation for a given posterior parameter sample. The code chunk below creates the function `my_post_predict()` which reads in separate values for each parameter in our model. Thus, this specific function is specific to our model setup. This also means that we will need to loop over all posterior samples in order to evaluate the posterior predictive *distribution*. The fourth argument to `my_post_predict()` is a posterior sample index to help with the book keeping. The last argument, `x_new`, is the grid of test points we wish to make predictions at.

``` r
my_post_predict <- function(b0_s, b1_s, sigma_s, s_id, x_new)
{
  # evaluate the linear predictor
  mu_s <- b0_s + b1_s * x_new
  
  # generate random observations
  y_s <- rnorm(n = length(x_new), mean = mu_s, sd = sigma_s)
  
  # package together
  tibble::tibble(
    x = x_new,
    mu = mu_s,
    y = y_s
  ) %>% 
    mutate(post_id = s_id)
}
```

As shown in the code chunk above, we will need to iterate over 4 separate variables to properly execute the `my_post_predict()` function. We can still use the `purrr` package to perform the looping for use by using the `pmap_*` family of functions. `pmap_*` generalizes the `map_*` and `map2_*` functions by allowing p-different variables to be iterated over.

In the code chunk below, `post_param_samples` is modified to include a specific column for the row ID. This row ID variable is passed into `my_post_predict()` as the `s_id` input argument. The `purrr::pmap_dfr()` function is then applied to iterate over all posterior samples.

``` r
post_param_samples_s <- post_param_samples %>% 
  tibble::rowid_to_column("post_id")

set.seed(4112)
post_preds <- purrr::pmap_dfr(list(post_param_samples_s$beta_0,
                                   post_param_samples_s$beta_1,
                                   post_param_samples_s$sigma,
                                   post_param_samples_s$post_id),
                              my_post_predict,
                              x_new = x_test)
```

The `post_preds` data object is structured in a long or tall style format to faciliate plotting in `ggplot2`. Calling `glimpse()` reveals the large number of rows contained in the dataset. **Can you think of why there are 250000 rows?**

``` r
post_preds %>% glimpse()
```

    ## Observations: 250,000
    ## Variables: 4
    ## $ x       <dbl> -3.00, -2.75, -2.50, -2.25, -2.00, -1.75, -1.50, -1.25...
    ## $ mu      <dbl> -3.71016579, -3.40489795, -3.09963011, -2.79436227, -2...
    ## $ y       <dbl> -3.9962713, -3.2827056, -2.9093437, -2.6124585, -2.708...
    ## $ post_id <int> 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...

What do you anticipate the posterior predictions look like? To get an idea, let's show the first 11 posterior predictions of ![\\mu](https://latex.codecogs.com/png.latex?%5Cmu "\mu").

``` r
post_preds %>% 
  filter(post_id %in% 1:11) %>% 
  ggplot(mapping = aes(x = x, y = mu)) +
  geom_line(mapping = aes(group = post_id)) +
  labs(y = expression(mu)) +
  theme_bw()
```

![](lecture_07_github_files/figure-markdown_github/viz_20_post_mu_preds-1.png)

To add some context, let's include the 10 training points used to fit the Bayesian model, as well as the *true* noise-free line. Does the figure below look famaliar?

``` r
post_preds %>% 
  filter(post_id %in% 1:11) %>% 
  ggplot(mapping = aes(x = x, y = mu)) +
  geom_line(mapping = aes(group = post_id)) +
  geom_point(data = train_df,
             mapping = aes(x = x, y = y),
             color = "red", size = 2) +
  geom_abline(slope = beta_1_true, intercept = beta_0_true,
              color = "red", linetype = "dashed", size = 1.05) +
  labs(y = expression(mu)) +
  theme_bw()
```

![](lecture_07_github_files/figure-markdown_github/viz_20_post_mu_preds_bb-1.png)

Let's now include the first 100 posterior lines, and set the transparency on each line to be 0.2. With just 100 lines, we should start to get a sense of the posterior uncertainty on the linear predictor, due to the *epistemic* uncertainty of the ![\\boldsymbol{\\beta}](https://latex.codecogs.com/png.latex?%5Cboldsymbol%7B%5Cbeta%7D "\boldsymbol{\beta}") parameters.

``` r
post_preds %>% 
  filter(post_id %in% 1:100) %>% 
  ggplot(mapping = aes(x = x, y = mu)) +
  geom_line(mapping = aes(group = post_id),
            alpha = 0.2) +
  geom_point(data = train_df,
             mapping = aes(x = x, y = y),
             color = "red", size = 2) +
  geom_abline(slope = beta_1_true, intercept = beta_0_true,
              color = "red", linetype = "dashed", size = 1.05) +
  labs(y = expression(mu)) +
  theme_bw()
```

![](lecture_07_github_files/figure-markdown_github/viz_50_post_mu_preds-1.png)

We have many more samples of the linear predictor we could include, but we will instead *summarize* the posterior samples of the linear predictor. The code chunk below calculates the posterior mean and the posterior 5th and 95th quantiles on the linear predictor. The posterior mean is displayed as a black line, and the posterior middle 90% *credible* interval is displayed as a ribbon. Based on the figure below, the posterior mean on ![\\mu](https://latex.codecogs.com/png.latex?%5Cmu "\mu") is very accurate relative to the *true* line.

``` r
post_preds %>% 
  group_by(x) %>% 
  summarise(avg_mu = mean(mu),
            q05_mu = quantile(mu, 0.05),
            q95_mu = quantile(mu, 0.95)) %>% 
  ungroup() %>% 
  ggplot(mapping = aes(x = x)) +
  geom_ribbon(mapping = aes(ymin = q05_mu,
                            ymax = q95_mu,
                            group = 1),
              fill = "grey") +
  geom_line(mapping = aes(y = avg_mu,
                          group = 1),
            color = "black", size = 1.15) +
  geom_point(data = train_df,
             mapping = aes(x = x, y = y),
             color = "red", size = 2) +
  geom_abline(slope = beta_1_true, intercept = beta_0_true,
              color = "red", linetype = "dashed", size = 1.05) +
  labs(y = expression(mu)) +
  theme_bw()
```

![](lecture_07_github_files/figure-markdown_github/viz_post_summary_mu-1.png)

The grey ribbon shosws uncertainty interval on the linear predictor. It is very important to remember, that is all that it shows. It is our middle 90% uncertainty interval on the **mean** trend with respect to ![x](https://latex.codecogs.com/png.latex?x "x"). Although the Bayesian term, *credible*, was used to describe this interval, it is also commonly referred to as a *confidence interval* on the mean. Regardless of the naming convention, it does **not** represent our uncertainty in potential response values. Uncertainty in predictions of observations, and not just means or averages or trends, require considering the **aleatory** source of uncertainty. Accounting for the additional *aleatory* uncertainty in a prediction is typically denoted by *prediction intervals*. You most likely have seen confidence intervals before, but prediction intervals are typically neglected.

To get used to dealing with random *aleatory* predictions, the code chunk below plots the randomly generated predictions associated with the first 11 posterior samples. Does the figure below look famaliar?

``` r
post_preds %>% 
  filter(post_id %in% 1:11) %>% 
  ggplot(mapping = aes(x = x, y = y)) +
  geom_point(color = "darkorange") + 
  theme_bw()
```

![](lecture_07_github_files/figure-markdown_github/viz_post_pred_rand_11-1.png)

For context, include the linear predictor posterior predictive summaries, to see where the random predictions "fall" relative to the mean trend.

``` r
post_preds %>% 
  group_by(x) %>% 
  summarise(avg_mu = mean(mu),
            q05_mu = quantile(mu, 0.05),
            q95_mu = quantile(mu, 0.95)) %>% 
  ungroup() %>% 
  ggplot(mapping = aes(x = x)) +
  geom_ribbon(mapping = aes(ymin = q05_mu,
                            ymax = q95_mu,
                            group = 1),
              fill = "grey") +
  geom_line(mapping = aes(y = avg_mu,
                          group = 1),
            color = "black", size = 1.15) +
  geom_point(data = post_preds %>% 
               filter(post_id %in% 1:11),
             mapping = aes(x = x, y = y),
             color = "darkorange", size = 2) +
  labs(y = "y") +
  theme_bw()
```

![](lecture_07_github_files/figure-markdown_github/viz_post_pred_rand_11_vs_mu-1.png)

Now summarize the posterior prediction interval as the middle 90% uncertainty interval, and include all 100 randomly generated observations from the *true* data generating process. As shown in the figure below, the posterior 90% prediction interval appears representative of the scatter in the observations.

``` r
post_preds %>% 
  group_by(x) %>% 
  summarise(avg_mu = mean(mu),
            q05_mu = quantile(mu, 0.05),
            q95_mu = quantile(mu, 0.95),
            q05_rand = quantile(y, 0.05),
            q95_rand = quantile(y, 0.95)) %>% 
  ungroup() %>% 
  ggplot(mapping = aes(x = x)) +
  geom_ribbon(mapping = aes(ymin = q05_rand,
                            ymax = q95_rand,
                            group = 1),
              fill = "darkorange") +
  geom_ribbon(mapping = aes(ymin = q05_mu,
                            ymax = q95_mu,
                            group = 1),
              fill = "grey") +
  geom_line(mapping = aes(y = avg_mu,
                          group = 1),
            color = "black", size = 1.15) +
  geom_point(data = demo_df,
             mapping = aes(x = x, y = y),
             color = "black", size = 1) +
  geom_point(data = train_df,
             mapping = aes(x = x, y = y),
             color = "red", size = 2) +
  geom_abline(slope = beta_1_true, intercept = beta_0_true,
              color = "red", linetype = "dashed", size = 1.05) +
  labs(y = "y") +
  theme_bw()
```

![](lecture_07_github_files/figure-markdown_github/viz_summarise_post_pred_rand_02-1.png)

In addition to visualizing trends, we can use the posterior predictive distribution to answer various questions about the response. For example, suppose we were interested in calculating the probability that the response, ![y](https://latex.codecogs.com/png.latex?y "y"), was between 0.25 and 1.15. Although we could consider answering that question with the observations directly, what if we were interested in that probability **as a function of the input x**? Although we have 100 observations in total for this example, we may not have enough to answer this type of question.

However, we can use the posterior predictive distribution to estimate the probability the response is within the desired interval with respect to ![x](https://latex.codecogs.com/png.latex?x "x"). Let's first plot the interval as horizontal lines on top of the posterior predictive distribution with respective to ![x](https://latex.codecogs.com/png.latex?x "x").

``` r
post_preds %>% 
  group_by(x) %>% 
  summarise(avg_mu = mean(mu),
            q05_mu = quantile(mu, 0.05),
            q95_mu = quantile(mu, 0.95),
            q05_rand = quantile(y, 0.05),
            q95_rand = quantile(y, 0.95)) %>% 
  ungroup() %>% 
  ggplot(mapping = aes(x = x)) +
  geom_ribbon(mapping = aes(ymin = q05_rand,
                            ymax = q95_rand,
                            group = 1),
              fill = "darkorange") +
  geom_ribbon(mapping = aes(ymin = q05_mu,
                            ymax = q95_mu,
                            group = 1),
              fill = "grey") +
  geom_line(mapping = aes(y = avg_mu,
                          group = 1),
            color = "black", size = 1.15) +
  geom_hline(yintercept = c(0.25, 1.15),
             color = "steelblue", size = 1.25) +
  labs(y = "y") +
  theme_bw()
```

![](lecture_07_github_files/figure-markdown_github/viz_post_pred_with_threshold_lines-1.png)

The probability of being within this defined interval is calculated and displayed with respect to ![x](https://latex.codecogs.com/png.latex?x "x") in the code chunk below.

``` r
post_preds %>% 
  group_by(x) %>% 
  summarise(prob_in_interval = mean(between(y, 0.25, 1.15))) %>% 
  ungroup() %>% 
  ggplot(mapping = aes(x = x, y = prob_in_interval)) +
  geom_line(mapping = aes(group = 1),
            color = "black", size = 1.2) +
  labs(y = "Pr(0.25 <= y <= 1.15)") +
  theme_bw()
```

![](lecture_07_github_files/figure-markdown_github/viz_prob_within_threshold-1.png)
