Predicting online shoppers’ purchasing intentions
================
Yazan Saleh
27/11/2020

-   [Summary](#summary)
-   [Introduction](#introduction)
-   [Methodology](#methodology)
    -   [Data](#data)
    -   [Analysis](#analysis)
-   [Results & Discussion](#results-discussion)
-   [References](#references)

``` r
library(here)
```

    ## Warning: package 'here' was built under R version 4.0.2

    ## here() starts at /Volumes/Data/study/ubc-mds/ubc-mds-github/block_3/dsci_522/labs/DSCI_522_group_31

``` r
library(tidyverse)
```

    ## Warning: package 'tidyverse' was built under R version 4.0.2

    ## ── Attaching packages ─────────────────────────────────────── tidyverse 1.3.0 ──

    ## ✓ ggplot2 3.3.2     ✓ purrr   0.3.4
    ## ✓ tibble  3.0.4     ✓ dplyr   1.0.2
    ## ✓ tidyr   1.1.2     ✓ stringr 1.4.0
    ## ✓ readr   1.4.0     ✓ forcats 0.5.0

    ## Warning: package 'ggplot2' was built under R version 4.0.2

    ## Warning: package 'tibble' was built under R version 4.0.2

    ## Warning: package 'tidyr' was built under R version 4.0.2

    ## Warning: package 'readr' was built under R version 4.0.2

    ## Warning: package 'dplyr' was built under R version 4.0.2

    ## ── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
    ## x dplyr::filter() masks stats::filter()
    ## x dplyr::lag()    masks stats::lag()

## Summary

In this project, we compare 3 different algorithms with the aim of
building a classification model to predict purchasing intentions of
online shoppers given browser session data and pageview data. Given our
dataset, random forest classifier was identified as the best model
compared to support vector machine and logistic regression classifiers.
Although the model performed relatively well in terms of accuracy with a
score 0.88, its performance was less robust when scored using `f1`
metric. Specifically, the model had an `f1` score of 0.66 and
mis-classified 376 observations, 131 of which were false negatives. The
131 incorrect classifications are significant as they represent
potential sources of missed revenue for e-commerce businesses.
Therefore, we recommend improving this model prior to deployment in the
real-world.

## Introduction

With the rising popularity of online shopping, particularly in the wake
of the 2020 coronavirus pandemic, there exists a strong growth
opportunity for businesses employing e-commerce solutions. While
increasing overall traffic to online stores is a critical first step,
higher traffic does not always convert into increased sales. Online
visitors may exit the site without making a purchase for a variety of
reasons including: loading times, website layout, pricing, and others.

In this project, we attempt to use machine learning to predict whether a
visitor of an online shopping website is intending on making a purchase
based on session data that looks at the pages they visit, the duration
of each visit, and the source of traffic among other factors.

Being able to predict purchasing intentions can be a valuable tool
because it could potentially allow businesses to implement measures to
better capture users with purchasing intentions and thus try to convert
them into actual revenue generators. For example, users with purchasing
intentions can be served with targeted content that aims to reduce
shopping cart abandonment. Predicting purchasing intentions can also be
useful for making more accurate forecasts about the business. A business
with a given portfolio of visitors may use the machine learning
algorithm to predict its revenue conversion rates which can be a
valuable metric in financial modeling.

## Methodology

### Data

The data set used in this project is the “Online Shoppers Purchasing
Intention” dataset provided by the [Gözalan
Group](http://www.gozalangroup.com.tr/) and used by Sakar et. al in
their analysis published in [Neural Computing and
Applications](https://link.springer.com/article/10.1007/s00521-018-3523-0)
\[@Sakar2019\]. The data set was sourced from UCI’s Machine Learning
Repository \[@Dua:2019\] at this
[link](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset).
The specific file used for the analysis found
[here](https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv).
Each row in the data set contains pageview and session information
belonging to a different visitor that browsed
[Columbia](https://www.columbia.com.tr), an online e-commerce platform
based in Turkey. Each row also contains the target class `REVENUE`, a
boolean flag indicating whether that user session contributed to revenue
or not. Pageview data includes the type of pages that the visitor
browsed to and the duration spent on each page. Session information
includes visitor-identifying data such as browser information, operating
system information as well visitor location and type.

High level information about the dataset variables can be found below:

| Index | Variable                 | Description                                                                                                                                                    |
|-------|--------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1     | Administrative           | Number of administrative pages visited                                                                                                                         |
| 2     | Administrative\_Duration | Duration of time spent on administrative pages                                                                                                                 |
| 3     | Informational            | Number of informational pages visited                                                                                                                          |
| 4     | Informational\_Duration  | Duration of time spent on informational pages                                                                                                                  |
| 5     | ProductRelated           | Number of product related pages visited                                                                                                                        |
| 6     | ProductRelated\_Duration | Duration of time spent on informational pages                                                                                                                  |
| 7     | BounceRate               | The percentage of visitors who enter the site from that page and then leave without triggering any other requests to the analytics server during that session. |
| 8     | ExitRate                 | The percentage that were the last in the session over all pageviews to the page                                                                                |
| 9     | PageValues               | The average value for a web page that a user visited before completing a transaction.                                                                          |
| 10    | SpecialDay               | How close the site visiting time is to a special day/holiday                                                                                                   |
| 11    | Month                    | Month of visiting time                                                                                                                                         |
| 12    | OperatingSystem          | OS used by the visitor                                                                                                                                         |
| 13    | Browser                  | Internet browser used by the visitor                                                                                                                           |
| 14    | Region                   | Demographic region when the user accessing the site                                                                                                            |
| 15    | TrafficType              | From which source the visitor arrived at the site                                                                                                              |
| 16    | VisitorType              | Visitor type as “New Visitor,” “Returning Visitor,” and “Other”                                                                                                |
| 17    | Weekend                  | If the visiting day is during weekend                                                                                                                          |
| 18    | Revenue                  | Prediction target                                                                                                                                              |

Values of feature 1-6 are derived from the visited pages’ URL and
reflected in real time when a user takes any actions on the site. Values
of feature 7-9: come from metrics measured by “Google Analytics” for
each page on the site.

Among those features, there are some features that might be important to
predict the target: ‘ProductRelated\_Duration,’ ‘BounceRates,’
‘ExitRates,’ ‘PageValues,’ ‘Month,’ ‘TrafficType’ and ‘VisitorType.’

### Analysis

Considering this is a binary classification problem, several algorithms
can be well-suited to the task. In our study, we compared 3 different
models, namely, support-vector machine (SVM), logistic regression, and
random forest classifier in their ability to classify visitors as a
revenue generator, (i.e. intending to purchase) or not. Initially all
features included in the data sets were used to fit and score the
models. Hyperparameters of each model were chosen based on `f1` scores
following randomized search cross-validation. Specifically, we tuned the
regularization and kernel coefficient for SVM; the regularization
hyperparameter for logistic regression; and the number of estimators and
maximum depth parameters for random forest classifier. The number of
folds used in cross-validation was 10 for both hyperparameter
optimization and model selection.

After selecting the best model with the best hyperparameters, we reduced
the number of features by eliminating the non-important ones using
recursive feature elimination (RFE).

The analysis was performed using the Python programming language
\[@Python\] along with the following packages: Altair
\[@vanderplas2018altair\], docopt \[@docopt\], feather \[@featherpy\],
knitr \[@knitr\], pandas \[@mckinney2010data\], and Scikit-learn
\[@pedregosa2011scikit\]. The code used to conduct this analysis can be
found [here](https://github.com/UBC-MDS/DSCI_522_group_31).

## Results & Discussion

Prior to fitting the model, we looked at how the distribution of each of
the features in the training set varies between the two classes (revenue
generator: orange, not a revenue generator: blue). This visualization
shows us overlap in the distribution of features across the two target
classes, although their spreads differ in some cases. As a result, we
opted to include all features in the initial analysis and subsequently
try to use RFE to better guide us at feature selection.

<div class="figure">

<img src="../img/eda/feature_density.png" alt="Figure.1 Density plots of numerical features by target class" width="80%" height="70%" />
<p class="caption">
Figure.1 Density plots of numerical features by target class
</p>

</div>

Following random search hyperparameter optimization and fitting on the
entire training dataset, random forest classifier with hyperparameters
`max_depth = 13` and `n_estimators = 65` was the best performing model
according to `f1` score.

``` r
comparison_table = read_csv("../data/processed/model_selection_result.csv")
```

    ## Warning: Missing column names filled in: 'X1' [1]

    ## 
    ## ── Column specification ────────────────────────────────────────────────────────
    ## cols(
    ##   X1 = col_double(),
    ##   index = col_character(),
    ##   fit_time = col_double(),
    ##   score_time = col_double(),
    ##   test_accuracy = col_double(),
    ##   test_f1 = col_double(),
    ##   test_recall = col_double()
    ## )

``` r
comparison_table <- comparison_table %>% select(-X1)
colnames(comparison_table)[1]="Model"
knitr::kable(comparison_table)
```

| Model               | fit\_time | score\_time | test\_accuracy | test\_f1 | test\_recall |
|:--------------------|----------:|------------:|---------------:|---------:|-------------:|
| Logistic Regression |    0.2141 |      0.0222 |         0.8548 |   0.6149 |       0.7554 |
| Random Forest       |    8.3427 |      0.1351 |         0.8841 |   0.6709 |       0.7702 |
| SVC                 |    7.7296 |      0.4416 |         0.8714 |   0.6374 |       0.7364 |

The model performance on the test set was less robust as `f1` score
dropped to 0.66 when considering our class of interest, presence of
revenue, as the positive class. Overall accuracy was relatively high at
0.88 although the model mis-classified 376 observations consisting of
245 false positives and 131 false negatives as per the confusion matrix
shown below.

<div class="figure">

<img src="../img/reports/confusion_matrix.png" alt="Figure.2 Confusion Matrix before Feature Selection" width="80%" height="70%" />
<p class="caption">
Figure.2 Confusion Matrix before Feature Selection
</p>

</div>

``` r
cr <- read_csv("../img/reports/classification_report.csv")
```

    ## Warning: Missing column names filled in: 'X1' [1]

    ## 
    ## ── Column specification ────────────────────────────────────────────────────────
    ## cols(
    ##   X1 = col_double(),
    ##   index = col_character(),
    ##   precision = col_double(),
    ##   recall = col_double(),
    ##   `f1-score` = col_double(),
    ##   support = col_double()
    ## )

``` r
cr <- cr %>% select(-X1)
colnames(cr)[1]="Class"
knitr::kable(cr)
```

| Class        | precision |    recall |  f1-score |      support |
|:-------------|----------:|----------:|----------:|-------------:|
| No-revenue   | 0.9525355 | 0.9051658 | 0.9282467 | 2594.0000000 |
| Revenue      | 0.6019417 | 0.7607362 | 0.6720867 |  489.0000000 |
| accuracy     | 0.8822575 | 0.8822575 | 0.8822575 |    0.8822575 |
| macro avg    | 0.7772386 | 0.8329510 | 0.8001667 | 3083.0000000 |
| weighted avg | 0.8969272 | 0.8822575 | 0.8876167 | 3083.0000000 |

We used recursive feature elimination (RFE) to attempt to achieve better
classification performance and identify the most important features. RFE
retained 37/74 features after transformation which came from 10 features
out of the original 17 as being most important to the classification
problem. Nevertheless, fitting the model on the new dataset that
includes these features only did not significantly affect performance as
shown below.

<div class="figure">

<img src="../img/reports/confusion_matrix_feature_selection.png" alt="Figure.3 Confusion Matrix after Feature Selection" width="80%" height="70%" />
<p class="caption">
Figure.3 Confusion Matrix after Feature Selection
</p>

</div>

``` r
cr <- read_csv("../img/reports/classification_report_feature_selection.csv")
```

    ## Warning: Missing column names filled in: 'X1' [1]

    ## 
    ## ── Column specification ────────────────────────────────────────────────────────
    ## cols(
    ##   X1 = col_double(),
    ##   index = col_character(),
    ##   precision = col_double(),
    ##   recall = col_double(),
    ##   `f1-score` = col_double(),
    ##   support = col_double()
    ## )

``` r
cr <- cr %>% select(-X1)
colnames(cr)[1]="Class"
knitr::kable(cr)
```

| Class        | precision |    recall |  f1-score |      support |
|:-------------|----------:|----------:|----------:|-------------:|
| No-revenue   | 0.9204254 | 0.9676176 | 0.9434317 | 2594.0000000 |
| Revenue      | 0.7640449 | 0.5562372 | 0.6437870 |  489.0000000 |
| accuracy     | 0.9023678 | 0.9023678 | 0.9023678 |    0.9023678 |
| macro avg    | 0.8422352 | 0.7619274 | 0.7936093 | 3083.0000000 |
| weighted avg | 0.8956216 | 0.9023678 | 0.8959045 | 3083.0000000 |

In the context of the model’s applicability, false negatives can be
argued to be more detrimental than false positives as they represent
untapped potential revenue sources. Therefore, we should prefer a model
with higher recall (sensitivty) over one with higher precision, which
was the case here. Nevertheless, and given the non-small number of false
negatives, our model might need further refinement before implementation
in the real-world.

One way of improving this model would be to employ oversampling to
address the class imbalance problem. Although we attempted to address
class imbalance by adjusting the weights associated with the target
class, Sakar et. al were able to achieve much better `f1` scores using
oversampled data sets and with similar algorithms (namely SVC and random
forest) \[@Sakar2019\]. Further, we could use a different method to
identify the most important features for this classification problem and
compare performance to the recursive feature elimination method we
employed here. Sakar et. al feature selection results include some
features from the same dataset, such as `Product related` and
`Administrative`, that were not identified by our RFE search. This
indicates that visiting `Product related` and `Administrative` pages is
important to the classification algorithm although our model and current
feature selection methodology was not able to identify that.

# References
