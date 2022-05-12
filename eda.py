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
        vax.fillna(0, inplace=True)
        vax[vax_cols_to_diff] = vax[vax_cols_to_diff].diff()

        ### STARTED PART 1 ###
        # daily mean daily cases + covid deaths for February 2021
        feb_2021 = cases.loc[
            (cases["date"].dt.month == 2) & (cases["date"].dt.year == 2021)
        ]
        feb_2021_mean_deaths = feb_2021["tot_deaths"].mean()
        feb_2021_mean_cases = feb_2021["tot_cases"].mean()

        print(cases.head())
        print(vax.head())
