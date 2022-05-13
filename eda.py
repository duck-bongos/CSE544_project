from datetime import datetime

import pandas as pd


def de_cumulate_column():
    pass


# our states are Massachussetts and Mississippi.
GROUP_11_STATES = ["MA", "MS"]


if __name__ in "__main__":
    cases_df = pd.read_csv(
        "United_States_COVID-19_Cases_and_Deaths_by_State_over_Time.csv"
    )
    vax_df = pd.read_csv("COVID-19_Vaccinations_in_the_United_States_Jurisdiction.csv")

    # do all this for each state Massachussetts first
    for state in GROUP_11_STATES:
        # grab the relevant state
        cases = cases_df[cases_df["state"] == state]
        cases = cases.rename(columns={"submission_date": "date"})
        vax = vax_df[vax_df["Location"] == state]
        vax = vax.rename(columns={"Location": "state", "Date": "date"})

        # update submission date to datetime for sorting
        cases["date"] = pd.to_datetime(cases["date"])
        vax["date"] = pd.to_datetime(vax["date"])

        # sort by date
        vax = vax.sort_values("date", ascending=True)
        cases = cases.sort_values("date", ascending=True)

        # cases columns to diff:
        cases_cols_to_diff = ["tot_cases", "conf_cases", "prob_cases"]
        cases.fillna(0, inplace=True)
        cases[cases_cols_to_diff] = cases[cases_cols_to_diff].diff()

        # vax columns to diff to de-cumulatize things
        vax_cols_to_diff = vax.columns[3:]
        vax[vax_cols_to_diff] = vax[vax_cols_to_diff].diff()

        """
        ## DATA CLEANING - MANDATORY TASK 1 ##
        NOTE TO ALL: The project prompt for this part says

        'Comment on your findings both for data cleaning
        (what issues you found, how you dealt with them) and outlier detection. 
        This will be 10% of the project grade.'
        
        I really didn't have any issues doing this. I either
        1. Am really good at this due to prior experience with data + pandas and 
        knew exactly what to do (great, but doesn't help us with this part of the report)
        2. Missed something by making an assumption I shouldn't have an need to fix this
        3. Absolutely, unequivocally biffed it.

        I mean it's weird how many rows have outliers for the vaccine data. might be worth looking into more.

        PLEASE TELL ME IF I NEED TO CHANGE ANYTHING.
        """

        # dropna() to drop any rows with NaN in them, apply to both datasets
        cases.dropna(inplace=True)
        vax.dropna(inplace=True)

        # worth resetting index at this point to make filtering easier
        cases = cases.reset_index()
        vax = vax.reset_index()

        # apply tukey's rule to each column, remove outliers that are non-zero values
        # transform each data column into a boolean column, take the all-true data
        """
        cases.describe() returns this matrix:
        -------------------------------------
                tot_cases     conf_cases   prob_cases      new_case  ...    conf_death   prob_death   new_death  pnew_death
        count    832.000000     832.000000   832.000000    832.000000  ...    832.000000   832.000000  832.000000  832.000000
        mean    2123.582933    1955.465144   168.117788   2121.533654  ...  13281.364183   344.465144   28.931490    1.367788
        std     4652.956193    6639.303039   403.000405   4653.389217  ...   6510.533026   251.101537   36.587088   11.631706
        min        0.000000  -96965.000000   -26.000000      0.000000  ...      0.000000     0.000000    0.000000   -3.000000
        25%      192.750000      99.750000     0.000000    192.750000  ...   8662.000000   205.000000    4.000000    0.000000
        50%     1073.500000     895.500000    39.500000   1060.000000  ...  16364.000000   335.000000   16.000000    0.000000
        75%     2087.750000    1881.000000   184.000000   2087.750000  ...  18346.000000   387.250000   38.250000    1.000000
        max    64715.000000  102762.000000  4505.000000  64715.000000  ...  22980.000000  1138.000000  209.000000  324.000000
        """

        # First quartile for each column
        cases_quantile_1 = cases.describe().iloc[4, :]

        # Third quartile for each column
        cases_quantile_3 = cases.describe().iloc[6, :]

        # Calculate interquartile range
        cases_iqr = cases_quantile_3 - cases_quantile_1

        # Calculate Tukey's rule  # formula is  x < Q1 - alpha * IQR || x > Q3 + alpha * IQR
        cases_low_outlier = cases_quantile_1 - (1.5 * cases_iqr)
        cases_high_outlier = cases_quantile_3 + (1.5 * cases_iqr)

        # repeat for vax data
        # First quartile for each column
        vax_quantile_1 = vax.describe().iloc[4, :]

        # Third quartile for each column
        vax_quantile_3 = vax.describe().iloc[6, :]

        # Calculate interquartile range
        vax_iqr = vax_quantile_3 - vax_quantile_1

        # Calculate Tukey's rule  # formula is  x < Q1 - alpha * IQR || x > Q3 + alpha * IQR
        vax_low_outlier = vax_quantile_1 - (1.5 * vax_iqr)
        vax_high_outlier = vax_quantile_3 + (1.5 * vax_iqr)

        # Find values that fall outside the range, excluding zeros
        # this is the slash-and-burn approach where if there are ANY outliers
        # within the row from any of the 10 numerical columns, drop the entire row

        # find all the row indices where there's an outlier
        outlier_case_idx = (cases[cases_low_outlier.index] >= cases_low_outlier).all(
            axis=1
        ) & ((cases[cases_high_outlier.index] <= cases_high_outlier).all(axis=1))

        # keep only rows where there are no outliers
        cases = cases[outlier_case_idx]

        # count the number of rows with at least 1 outlier in it:
        print(
            f"Rows with outliers in case data: {len(outlier_case_idx) - outlier_case_idx.astype(int).sum()}"
        )
        print(
            f"% Rows with outliers in vaccine data: {(len(outlier_case_idx) - outlier_case_idx.astype(int).sum())/ len(outlier_case_idx)}"
        )

        # find all the row indices where there's an outlier
        outlier_vax_idx = (vax[vax_low_outlier.index] >= vax_low_outlier).all(
            axis=1
        ) & ((vax[vax_high_outlier.index] <= vax_high_outlier).all(axis=1))

        print(
            f"Rows with outliers in vaccine data: {len(outlier_vax_idx) - outlier_vax_idx.astype(int).sum()}"
        )
        print(
            f"% Rows with outliers in vaccine data: {(len(outlier_vax_idx) - outlier_vax_idx.astype(int).sum())/ len(outlier_vax_idx)}"
        )

        # keep only rows where there are no outliers
        vax = vax[outlier_vax_idx]

        """Slash and burn, keep zero values. I admit, I, Dan Billmann, 
        have not tested the second half of this out.

        # it's either within the range (both higher AND (&) lower)
        # OR it's outside the range, ((either higher OR lower) AND equal to zero)
        ((cases[cases_low_outlier.index] >= cases_low_outlier).all(axis=1) & 
        (cases[cases_high_outlier.index] <= cases_high_outlier).all(axis=1)) |
        ((cases[cases_low_outlier.index] <= cases_low_outlier).all(axis=1) |
        (cases[cases_high_outlier.index] >= cases_high_outlier).all(axis=1)) &
        (cases[cases_high_outlier.index] == 0)
        """
        print()

        ### STARTED ANALYSIS PART 1 ###
        # daily mean daily cases + covid deaths for February 2021
        feb_2021 = cases.loc[
            (cases["date"].dt.month == 2) & (cases["date"].dt.year == 2021)
        ]
        feb_2021_mean_deaths = feb_2021["tot_deaths"].mean()
        feb_2021_mean_cases = feb_2021["tot_cases"].mean()

        ### STARTED PART 2 ###
        print(cases.head())
        print(vax.head())
