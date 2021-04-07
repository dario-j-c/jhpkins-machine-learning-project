Project\_Report
================
Dario J C
04/04/2021

``` r
library(tidyverse)
library(tidymodels)

library(workflowsets)
```

    ## Warning: package 'workflowsets' was built under R version 4.0.4

``` r
library(stacks)
```

    ## Warning: package 'stacks' was built under R version 4.0.4

``` r
set.seed(123)
```

## Executive Summary

This project will be exploring the provided Weight Lifting Exercises
Dataset which captures movement data while carrying out specific
exercises.

The goal was to predict the outcome variable “**classe**” using any
chosen predictor variables and methodology.

I chose to approach the problem using gradient boosted trees and
decision trees while retaining almost all of the attributes to be used
as dependants. This was done as these methods are useful for
classification problems while sidestepping the need for a large amount
of preprocessing (e.g. normalizing or scaling the data).

The data was also cross validated using stratified sampling where the
stratas were based on the outcome variable. This was to ensure each
stata (fold) represented the outcome equally and allowed for further
protection from overfitting. Other steps were also tested and mentioned
in the document.

I concluded by choosing a gradient boosted trees model which gave me
accuracy of approximately 97% on the training data. As I used cross
validation, hopefully I will not see a significant drop in accuracy when
exposed to novel data, in other words, I don’t believe my out of sample
error will be large.

A prediction for the testing data was also done.

You will notice I chose to use the `tidymodels` package instead of
`caret` to carry out my analysis. In some ways you may see see
`tidymodel` as `caret`‘s spiritual successor since the same individual,
Max Kuhn (along with others) is the author of the package, where
Mr. Kuhn has designed it to be a ’tidy’ version of `caret`. Personally
I’m simply more familiar with it. The text blurbs and comments will
explain what the code is doing to allow for easier reading.

As I’m now learning R and code in general, I’d appreciate any advice or
corrections.

## Preliminaries

I download the given data set.

``` r
# Download the data

pml_training <- read_csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",
                         col_types = cols(raw_timestamp_part_1 = col_double(),
                                          cvtd_timestamp = col_datetime(format = "%d/%m/%Y %H:%M"),
                                          kurtosis_picth_belt = col_double(),
                                          kurtosis_roll_belt = col_double(),
                                          kurtosis_yaw_belt = col_double(),
                                          skewness_roll_belt = col_double(),
                                          skewness_roll_belt.1 = col_double(),
                                          skewness_yaw_belt = col_double(),
                                          max_yaw_belt = col_double(),
                                          min_yaw_belt = col_double(),
                                          amplitude_yaw_belt = col_double(),
                                          kurtosis_picth_arm = col_double(),
                                          kurtosis_yaw_arm = col_double(),
                                          skewness_pitch_arm = col_double(),
                                          skewness_yaw_arm = col_double(),
                                          kurtosis_yaw_dumbbell = col_double(),
                                          skewness_yaw_dumbbell = col_double(),
                                          kurtosis_roll_forearm = col_double(),
                                          kurtosis_picth_forearm = col_double(),
                                          kurtosis_yaw_forearm = col_double(),
                                          skewness_roll_forearm = col_double(),
                                          skewness_pitch_forearm = col_double(),
                                          skewness_yaw_forearm = col_double(),
                                          max_yaw_forearm = col_double(),
                                          min_yaw_forearm = col_double(),
                                          amplitude_yaw_forearm = col_double()))
```

The data was cleaned by removing any columns with NAs and any columns
with just one variable. Any columns with only text were switched from
character to factor.

``` r
# Remove columns with any NA's from data frame
rev_pml_training <- Filter(function(x)!any(is.na(x)), pml_training)
# Remove columns with only one factor i.e. all values are the same
rev_pml_training <- Filter(function(x)length(unique(x)) > 1, rev_pml_training)
# Change remaining columns which are characters to factors
rev_pml_training <- rev_pml_training %>% 
  mutate_if(is.character, as.factor)
```

The stratified folds were created using the cleaned dataset

``` r
folds <- vfold_cv(rev_pml_training,
                  strata = classe,    # This is stratified sampling
                  v = 5)              # This is 5-fold cross-validation
```

## Recipes

Recipes are `tidymodel`’s way for specifying the steps to apply to a
dataset before applying a model to it. Although they can specify both
preprocessing and processing steps, I will be using them for
preprocessing only. This allowed me to use this one main recipe for both
of the models I used.

You will notice at this step I already specified the formula to be used
(`classe` is the outcome variable and every other column, excluding
seven (7) I specify, are to be predictors).

``` r
# This is the recipe to be used with trees
classe_rec <- recipe(classe ~ .,
                     data = rev_pml_training) %>%
# Update the first seven (7) columns as ID and thus not use them as predictors
  update_role(X1,
              user_name,
              raw_timestamp_part_1,
              raw_timestamp_part_2,
              cvtd_timestamp,
              new_window,
              num_window,
              new_role = "ID") %>%
  # Remove all predictors which only have a single value
  step_zv(all_predictors())
```

### Filters

Filters are also recipes but they are used to specify steps to be tested
to see how effective they are for your chosen metric(s).

In this case I’ve created two filters,

-   one filter (`step_corr`) attempts to remove variables to keep the
    largest absolute correlation between variables beneath a threshold
    you specify.

-   The second filter (`step_pca`) also attempts to limit inter-variable
    correlation but through the use of principal component analysis.

These methods also allowed me to set hyperparamters for them (as well as
in the following code) by using the placeholder `tune()`. `tidymodel`
will set these parameters for you while using your specified metric(s)
as the goal post.

``` r
filter_cor_rec <- 
   classe_rec %>%
   step_corr(all_predictors(),
             threshold = tune())
```

``` r
filter_pca_rec <- 
   classe_rec %>% # classe_knn_rec
   step_pca(all_predictors(), num_comp = tune()) %>% 
   step_normalize(all_predictors())
```

## Models

Models are used to specify the model used and formulas, in this case I
chose to only specify the model for cleaner code.

Although using different packages and models, you can see that
specifying the code uses the same terms. This is because `tidymodel`
keeps care of interacting with the different packages.

Here I specify the two models, to be used and choose which of their
hyperparameters to be tuned.

``` r
# Decision trees
dtree_mod <- decision_tree(tree_depth = tune(),
                    cost_complexity = tune(),
                    min_n = tune()) %>%
  set_engine("rpart") %>%
  set_mode("classification") %>%
  translate()

# Boosted trees
xgb_mod <- boost_tree(tree_depth = tune(),
                      learn_rate = tune(),
                      loss_reduction = tune(),
                      min_n = tune(),
                      sample_size = tune(),
                      trees = 1000) %>%
  set_engine("xgboost") %>%
  set_mode("classification") %>%
  translate()
```

## Workflow

Workflows allow me to combine the components I created and combine them
as I see fit.

Here my code is set up to model the following:

-   Boosted Trees

    -   WIth PCA decorrelation

    -   With decorrelation via variable removal

    -   As is

-   Decision Trees

    -   With PCA decorrelation

    -   With decorrelation via variable removal

    -   As is

``` r
# Create Workflow sets
all_wfset <-
  workflow_set(
    preproc = list(simple = classe_rec,
                   filter_cor = filter_cor_rec,
                   filter_pca = filter_pca_rec),
    models = list(decision_tree = dtree_mod,
                  boosted_trees = xgb_mod),
    cross = TRUE)
```

### Tuning

I also tune my values to find the best hyperparameters for my data. Here
I use a grid of five (5).

``` r
# Bear in mind this code snippet takes some time to run, as such I made preparations to avoid the long run time

grid_ctrl <-
  control_grid(
    save_pred = TRUE,
    parallel_over = "everything",
    save_workflow = TRUE,
    verbose = TRUE
    )

# The if statement is a quick work around for the time needed to tune
if (any(list.files() == "grid.results.RData")) {
  load("grid.results.RData")
  } else {
    grid_results <-
      all_wfset %>%
      workflow_map(fn = "tune_grid",
                   seed = 1503,
                   resamples = folds,
                   grid = 5,
                   control = grid_ctrl,
                   verbose = TRUE
               )
    save(grid_results, file = "grid.results.RData")
  }

# The total number of models ran
num_grid_models <- nrow(collect_metrics(grid_results,
                                        summarize = FALSE))
```

In total, I generated 300 models (including resampling & tuning) and
will choose the best one to be used for predictions.

## Results

I found the best model was **boosted trees using decorrelation via
removing variables**, details can be found in the following table and
plot.

Interestingly, decorrelation using PCA proved to be worse than using no
decorrelation at all, rather than assume this to be a fault of PCA, I’m
currently leaning to its under performance being due to my ignorance in
how to best use PCA in general.

I’d also note that the decision trees performed better without
decorrelation and considering how much faster I personally found the
decision tree to run than gradient boosted trees with only a small drop
in accuracy, if time constrained I would use a decision tree.

``` r
# Table showing best performing models
ranked_results <- grid_results %>%
  rank_results(select_best = TRUE) %>%
  select(rank,
         .config,
         wflow_id,
         metric = .metric,
         mean) %>%
 slice_head(n = 12)
ranked_results
```

    ## # A tibble: 12 x 5
    ##     rank .config              wflow_id                 metric    mean
    ##    <int> <chr>                <chr>                    <chr>    <dbl>
    ##  1     1 Preprocessor2_Model1 filter_cor_boosted_trees accuracy 0.970
    ##  2     1 Preprocessor2_Model1 filter_cor_boosted_trees roc_auc  0.998
    ##  3     2 Preprocessor1_Model1 simple_boosted_trees     accuracy 0.935
    ##  4     2 Preprocessor1_Model1 simple_boosted_trees     roc_auc  0.994
    ##  5     3 Preprocessor1_Model2 simple_decision_tree     accuracy 0.886
    ##  6     3 Preprocessor1_Model2 simple_decision_tree     roc_auc  0.974
    ##  7     4 Preprocessor4_Model1 filter_cor_decision_tree accuracy 0.765
    ##  8     4 Preprocessor4_Model1 filter_cor_decision_tree roc_auc  0.931
    ##  9     5 Preprocessor2_Model1 filter_pca_boosted_trees accuracy 0.569
    ## 10     5 Preprocessor2_Model1 filter_pca_boosted_trees roc_auc  0.847
    ## 11     6 Preprocessor3_Model2 filter_pca_decision_tree accuracy 0.482
    ## 12     6 Preprocessor3_Model2 filter_pca_decision_tree roc_auc  0.772

``` r
# Plot comparing metrics for the best models 
autoplot(
  grid_results,
  select_best = TRUE)
```

![](ml_project_files/figure-gfm/Display%20Results-1.png)<!-- -->

``` r
# Plot AUC-ROC curve for the best model
grid_results %>%
  collect_predictions(summarise = TRUE) %>%
  filter(.config == as.character(ranked_results[1,".config"]) &
           wflow_id == as.character(ranked_results[1,"wflow_id"])) %>%
  roc_curve(classe, .pred_A:.pred_E) %>%
  ggplot(aes(x = 1 - specificity, y = sensitivity)) +
  geom_point(lwd = 0.5, alpha = 0.4) +
  geom_abline(lty = 3) +
  coord_equal() +
  labs(title = "AUC-ROC Curve for best model")
```

![](ml_project_files/figure-gfm/Display%20Results-2.png)<!-- -->

## Predictions

I will now apply the best model generated to predict on the test data.
You will notice I will only now import the data, thus allowing me to
avoid tampering with it.

``` r
# Download Test Data in same the manner as Training Data
pml_test <- read_csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",
                         col_types = cols(raw_timestamp_part_1 = col_double(),
                                          cvtd_timestamp = col_datetime(format = "%d/%m/%Y %H:%M"),
                                          kurtosis_picth_belt = col_double(),
                                          kurtosis_roll_belt = col_double(),
                                          kurtosis_yaw_belt = col_double(),
                                          skewness_roll_belt = col_double(),
                                          skewness_roll_belt.1 = col_double(),
                                          skewness_yaw_belt = col_double(),
                                          max_yaw_belt = col_double(),
                                          min_yaw_belt = col_double(),
                                          amplitude_yaw_belt = col_double(),
                                          kurtosis_picth_arm = col_double(),
                                          kurtosis_yaw_arm = col_double(),
                                          skewness_pitch_arm = col_double(),
                                          skewness_yaw_arm = col_double(),
                                          kurtosis_yaw_dumbbell = col_double(),
                                          skewness_yaw_dumbbell = col_double(),
                                          kurtosis_roll_forearm = col_double(),
                                          kurtosis_picth_forearm = col_double(),
                                          kurtosis_yaw_forearm = col_double(),
                                          skewness_roll_forearm = col_double(),
                                          skewness_pitch_forearm = col_double(),
                                          skewness_yaw_forearm = col_double(),
                                          max_yaw_forearm = col_double(),
                                          min_yaw_forearm = col_double(),
                                          amplitude_yaw_forearm = col_double()))
```

    ## Warning: Missing column names filled in: 'X1' [1]

``` r
# Keep only the columns found in Training Data
rev_pml_test <- pml_test[intersect(names(rev_pml_training), names(pml_test))] %>%
  mutate_if(is.character, as.factor)

# Finalise model

#Extract tuned parameters for best model
best_result <- grid_results %>%
  pull_workflow_set_result(
    as.character(ranked_results[1,"wflow_id"])
    ) %>%
  select_best(metric = "accuracy")


# The if statement is a quick work around for the time needed to calculate
if (any(list.files() == "final.fit.RData")) {
  load("final.fit.RData")
  } else {
    final_fit <- grid_results %>%
      pull_workflow(
        as.character(ranked_results[1,"wflow_id"])
        ) %>%
      finalize_workflow(best_result) %>%
      fit(rev_pml_training)
    
    save(final_fit, file = "final.fit.RData")
  }


# Predict on test data using model
prediction <- predict(final_fit, rev_pml_test)
prediction
```

    ## # A tibble: 20 x 1
    ##    .pred_class
    ##    <fct>      
    ##  1 B          
    ##  2 A          
    ##  3 B          
    ##  4 A          
    ##  5 A          
    ##  6 C          
    ##  7 D          
    ##  8 B          
    ##  9 A          
    ## 10 A          
    ## 11 B          
    ## 12 C          
    ## 13 B          
    ## 14 A          
    ## 15 E          
    ## 16 E          
    ## 17 A          
    ## 18 B          
    ## 19 B          
    ## 20 B
