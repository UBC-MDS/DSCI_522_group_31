Predicting online shoppers’ purchasing intentions
================
Yazan Saleh
27/11/2020

The model performance on the test set was less robust as `f1` score
dropped to 0.67 when considering our class of interest, presence of
revenue, as the positive class. Overall accuracy was relatively high at
0.88 (Table 3) although the model mis-classified 363 observations
consisting of 246 false positives and 117 false negatives as per the
confusion matrix shown below (Figure 3).

<div style="padding: 20px; overflow: hidden;font-size:90%">

<div
style="padding:10px; width: 40%;float: right; padding-bottom: 100%;margin-bottom: -100%;">

<div class="figure">

<img src="../img/reports/confusion_matrix.png" alt="&lt;b&gt;Figure.3 Confusion Matrix before Feature Selection. High number of false positive results.&lt;/b&gt;" width="100%" height="40%" />
<p class="caption">
<b>Figure.3 Confusion Matrix before Feature Selection. High number of
false positive results.</b>
</p>

</div>

</div>

<div
style="width: 60%; float: left; padding-bottom: 100%; margin-bottom: -100%; ">

| Class        | precision |    recall |  f1-score |      support |
|:-------------|----------:|----------:|----------:|-------------:|
| No-revenue   | 0.9525355 | 0.9051658 | 0.9282467 | 2594.0000000 |
| Revenue      | 0.6019417 | 0.7607362 | 0.6720867 |  489.0000000 |
| accuracy     | 0.8822575 | 0.8822575 | 0.8822575 |    0.8822575 |
| macro avg    | 0.7772386 | 0.8329510 | 0.8001667 | 3083.0000000 |
| weighted avg | 0.8969272 | 0.8822575 | 0.8876167 | 3083.0000000 |

**Table 3: Classification report of the best model before applying
feature selection. High accuracy but low f1 and recall score for the
positive class.**

</div>

</div>
