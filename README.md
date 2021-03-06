# CSE544_project
Repository for SBU CSE544 project. 

# Things Dan already did: 
I already wrote a permutation test function and put it in the `utilities.py` file. You can import it to any other `.py` file by
writing `from .utilities import permutation_test`.

# Tasks
## Cases Dataset
1. In this step, we want to check, for both the states assigned to your group,
how the mean of monthly COVID19 stats has changed between Feb 2021 and March
2021. Apply the Wald’s test, Z-test, and t-test (assume all are applicable) to
check whether the mean of COVID19 deaths and #cases are different for Feb’21
and March’21 in the two states. That is, we are checking, for each state
separately, whether the mean of daily cases and the mean of daily deaths for
Feb’21 is different from the corresponding mean of daily values for March’21.
Use MLE for Wald’s test as the estimator; assume for Wald’s estimator purposes
that daily data is Poisson distributed. Note, you have to report results for
deaths and #cases in both states separately. After running the test and
reporting the numbers, check and comment on whether the tests are applicable or
not. First use one-sample tests for Wald’s, Z-test, and t-test by computing the
sample mean of daily values from Feb’21 and using that as a guess for mean of
daily values for March’21; here, your sample data for computing sample mean
will be the 28 daily values in Feb’21 whereas your sample data for running the
test will be the 31 daily values of March’21. Then, repeat with the two-sample
version of Wald’s and two-sample unpaired t-test (here, your two samples will
be the 28 values of Feb’21 and the 31 values of March’21). Use α=0.05 for all.
For t-test, the threshold to check against is tn-1,α/2 for two-tailed, where n
is the number of data points. You can find these values in online t tables,
similar to z tables. For Z-test, use the uncorrected sample standard deviation
(n in the denominator instead of n-1) of the entire COVID19 dataset you have
for each state as the true sigma value.
2. Inference the equality of distributions between the two states (distribution
of daily #cases and daily #deaths) for the last three months of 2021 (Oct, Nov,
Dec) of your dataset using K-S test and Permutation test. For the K-S test, use
both 1-sample and 2-sample tests. For the 1-sample test, try Poisson,
Geometric, and Binomial. To obtain parameters of these distributions to check
against in 1-sample KS, use MME on the Oct-Dec 2021 data of the first state in
your dataset to obtain parameters of the distribution, and then check whether
the Oct-Dec 2021 data for the second state in your dataset has the distribution
with the obtained MME parameters. For the permutation test, use 1000 random
permutations. Use a threshold of 0.05 for both K-S test and Permutation test.
3. For this task, sum up the daily stats (cases and deaths) from the two states
assigned to you. Assume day 1 is June 1st 2020. Assume the combined daily
deaths are Poisson distributed with parameter λ. Assume an Exponential prior
(with mean β) on λ. Assume β = λMME where the MME is found using the first four
weeks data (so the first 28 days of June 2020) as the sample data. Now, use the
fifth week’s data (June 29 to July 5) to obtain the posterior for λ via
Bayesian inference. Then, use the sixth week’s data to obtain the new
posterior, using prior as posterior after week 5. Repeat till the end of week 8
(that is, repeat till you have posterior after using 8th week’s data). Plot all
posterior distributions on one graph. Report the MAP for all posteriors.

## Vaccinations Dataset
4. In this task, we want to predict #vaccines for each state. Use the
Vaccinations dataset to predict the COVID19 #vaccines administered for the
fourth week in May 2021 using data from the first three weeks of May 2021. Do
this separately for each of the three states. Use the following four prediction
techniques: (i) AR(3), (ii) AR(5), (iii) EWMA with alpha = 0.5, and (iv) EWMA
with alpha = 0.8. Report the accuracy (MAPE as a % and MSE) of your predictions
using the actual fourth week data. 
5. Use the paired T-test to determine the equality of means of the #vaccines
administered between the two states for the months of September 2021 and
November 2021. Report your observations and plausible explanations for the
observations. Use 0.05 level of significance and uncorrected sample standard
deviation for the purpose of this test.  You can assume the test to be valid
for this part.
