"""For part 3, Exploratory Data Analysis. 

Part A: 
Gun violence towards others decreased during COVID: there is an inverse relationship between 
COVID cases and violent gun incidents.

Part B:
Check if gun violence changed after some local event or rule

"""
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
from utilities import (
    pearsons_correlation_coefficient,
    plot_cases_and_gun,
    scale_dataset,
)
from vax import *

# Use this arbitrary value for pearson correlation analysis
PEARSON_CRITICAL_VALUE = 0.5


def get_diffs(df: pd.DataFrame):
    df[["cases", "deaths"]] = df[["cases", "deaths"]].diff()
    return df


def part_1(cases: pd.DataFrame, gun_vals: pd.DataFrame):
    # run pearson correlation
    """To run a Pearson's Correlation test, the number of samples must be the same
    for X and Y RVs. There are more samples of gun violence (1746) data than cases (813). So we
    need to randomly sample 813 observations from gun violence many times (1000) to test the
    hypothesis properly.
    """
    rhos_injured = np.zeros(1000)
    rhos_killed = np.zeros(1000)

    # This part is non-deterministic, may or may not achieve the same results.
    for i in range(len(rhos_killed)):
        gun_killed_subset = np.random.choice(gun_vals["killed"].values, size=len(cases))

        rhos_killed[i] = pearsons_correlation_coefficient(
            scale_dataset(gun_killed_subset), scale_dataset(cases["cases"].values)
        )

        gun_injured_subset = np.random.choice(gun_vals["injured"], size=len(cases))
        rhos_injured[i] = pearsons_correlation_coefficient(
            scale_dataset(gun_injured_subset), scale_dataset(cases["cases"].values)
        )

    avg_rho_injured = sum(rhos_injured) / len(rhos_injured)
    avg_rho_killed = sum(rhos_killed) / len(rhos_killed)

    print("Pearson Correlation for violent gun injuries and COVID cases.")
    reject = "reject" if avg_rho_injured > PEARSON_CRITICAL_VALUE else "do not reject"
    print(
        f"With a rho value of {avg_rho_injured}, we {reject} the null hypothesis that gun violence and the COVID-19 pandemic are correlated.\n"
    )

    print("Pearson Correlation for violent gun deaths and COVID cases.")
    reject = "reject" if avg_rho_killed > PEARSON_CRITICAL_VALUE else "do not reject"
    print(
        f"With a rho value of {avg_rho_killed}, we {reject} the null hypothesis that gun violence and the COVID-19 pandemic are correlated.\n"
    )


def part_2():
    pass


if __name__ in "__main__":
    cases = pd.read_csv(
        "United_States_COVID-19_Cases_and_Deaths_by_State_over_Time.csv"
    )
    vax = pd.read_csv("COVID-19_Vaccinations_in_the_United_States_Jurisdiction.csv")
    gun = pd.read_csv("US-Gun-Violence.csv")

    cases = cases.rename(
        columns={"submission_date": "date", "tot_cases": "cases", "tot_death": "deaths"}
    )
    cases["date"] = pd.to_datetime(cases["date"])
    cases.sort_values("date", inplace=True)

    # remove data after 2021 because that's where our gun data stops
    cases = cases[cases["date"] <= datetime(2021, 12, 31)]

    # get daily differences in cases and deaths across the country
    cases = cases.groupby("date").apply(sum)[["cases", "deaths"]].diff()
    cases.iloc[0, :] = 0.0

    vax = vax.rename(
        columns={"Location": "state", "Date": "date", "Administered": "admin"}
    )
    vax["date"] = pd.to_datetime(vax["date"])
    # select out just date, admin, and state field
    # sort by date
    vax = vax[["date", "state", "admin"]].sort_values("date", ascending=True)

    gun = gun.rename(columns={"incident_date": "date", "city_or_county": "region"})
    gun = gun[["date", "killed", "injured"]]

    # ensure the dates for gun are the same as dates for cases
    # earliest_case_date = cases["date"].min()
    gun["date"] = pd.to_datetime(gun["date"])
    # gun = gun[(gun["date"] >= earliest_case_date)]

    # aggregate violent gun across the country incidents by date
    gun = gun.groupby("date").apply(sum)[["killed", "injured"]].reset_index()
    gun.sort_values("date", inplace=True)

    # ASSUMPTION: for days without gun violence reported, fill in a zero.
    # We do this for a few reasons:
    # 1. Matches the cases dataset.
    # 2. Ease of plotting cases vs gun violence.
    # 3. Naive optimism that nobody in the U.S. got shot that day.
    gun = gun.set_index("date")
    # fill in missing dates with NaN
    gun = gun.asfreq("D")
    no_gun_data = gun.isna().all(axis=1)

    # only keep values where gun violence is reported
    gun_vals = gun[~no_gun_data]
    cases_vals = cases[~no_gun_data]

    # Optional: plot_cases_and_gun(cases, gun.fillna(0))

    print("Part 1")
    part_1(cases_vals, gun_vals)

    part_2()
