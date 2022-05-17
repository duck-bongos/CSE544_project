"""For part 3, Exploratory Data Analysis. 

Part A: 
    Hypothesis 1: Gun violence towards others decreased during COVID: there is an inverse relationship between 
                  COVID cases and violent gun incidents.
    
    Hypothesis 2: 
    During Covid, is there a correlation between day of the week and whether COVID or 
    guns are a bigger KILLER. Run chi squared on this. 

Part B:
Check if gun violence changed after some local event or rule
 - George Floyd?

"""
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as ss
from utilities import (
    pearsons_correlation_coefficient,
    plot_cases_and_gun,
    scale_dataset,
    ks_test2,
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

    # Optional:
    # plot_cases_and_gun(cases, gun.fillna(0), fname="gun_violence_vs_covid_cases.png")

    print("Part 1")
    part_1(cases_vals, gun_vals)

    """Hypothesis 2: 
    During Covid, is there a correlation between day of the week and whether COVID or 
    guns are a bigger KILLER in NY. Run chi squared on this."""
    gun = pd.read_csv("US-Gun-Violence.csv")
    gun = gun[gun["state"] == "NY"]
    gun = gun.rename(columns={"incident_date": "date", "city_or_county": "region"})
    gun = gun[["date", "killed", "injured"]]

    cases = pd.read_csv(
        "United_States_COVID-19_Cases_and_Deaths_by_State_over_Time.csv"
    )
    cases = cases[cases["state"] == "NY"]
    cases = cases.rename(
        columns={"submission_date": "date", "tot_cases": "cases", "tot_death": "deaths"}
    )
    cases["date"] = pd.to_datetime(cases["date"])
    cases.sort_values("date", inplace=True)

    # remove data after 2021 because that's where our gun data stops
    cases = cases[cases["date"] <= datetime(2021, 12, 31)]

    cases = cases.set_index("date")

    # ensure the dates for gun are the same as dates for cases
    # earliest_case_date = cases["date"].min()
    gun["date"] = pd.to_datetime(gun["date"])

    # focus on covid dates
    gun_vals = gun_vals[gun_vals.index >= cases.index.min()]

    # pandas .weekday() on date. 0 is Monday, 6 is Sunday
    gun_vals["weekday"] = gun_vals.index
    gun_vals["weekday"] = gun_vals["weekday"].apply(lambda x: x.weekday())

    deaths = gun_vals[["killed", "weekday"]]
    deaths = deaths.rename(columns={"killed": "gunDeaths"})
    deaths["covidDeaths"] = cases_vals["deaths"]
    deaths["more_guns_1_more_covid_0"] = (
        gun_vals["killed"] >= cases_vals["deaths"]
    ).astype(int)

    # bucket counts
    weekday_bucket_percentages = deaths["weekday"].value_counts() / len(deaths)

    death_type_bucket = deaths["more_guns_1_more_covid_0"].value_counts() / len(deaths)

    # compute q_obs
    q_obs_gun = {}
    q_obs_gun[1] = 0
    q_obs_gun[5] = 0
    q_obs_cov = {}
    q_obs_cov[1] = 530
    q_obs_cov[5] = 530

    for u in deaths.weekday.unique():
        q_obs_gun[u] = len(
            deaths[(deaths.weekday == u) & deaths["more_guns_1_more_covid_0"] == 1]
        )
        q_obs_cov[u] = len(
            deaths[(deaths.weekday == u) & deaths["more_guns_1_more_covid_0"] == 0]
        )

    wd = pd.DataFrame(q_obs_gun.items(), columns=["weekday", "count"])
    wd = wd.set_index("weekday")
    cd = pd.DataFrame(q_obs_cov.items(), columns=["weekday", "count"])
    cd = cd.set_index("weekday")
    d = pd.merge(wd, cd, left_index=True, right_index=True)

    tot_x = d["count_x"].sum()
    exp_x = tot_x / len(d)
    tot_y = d["count_y"].sum()
    exp_y = tot_y / len(d)

    # calculate q_obs
    d["q_obs_x"] = (exp_x - d["count_x"]) ** 2 / exp_x
    d["q_obs_y"] = (exp_y - d["count_y"]) ** 2 / exp_y

    # E
    Q_Obs = d[["q_obs_x", "q_obs_y"]].sum().sum()
    print(Q_Obs)

    # calculate dfs
    deg_free = (len(d) - 1) ** 2

    p_value = ss.chi2.cdf(Q_Obs, deg_free)
    print(p_value)

    # HYPOTHESIS 3
    # what % of injuries are fatal:
    gun = pd.read_csv("US-Gun-Violence.csv")

    gun = gun.rename(columns={"incident_date": "date", "city_or_county": "region"})
    gun = gun[["date", "killed", "injured"]]
    gun["date"] = pd.to_datetime(gun["date"])
    gun = gun.set_index("date")

    gun["fatal_incident_rate"] = gun["killed"] / (gun["killed"] + gun["injured"])

    vax = pd.read_csv("COVID-19_Vaccinations_in_the_United_States_Jurisdiction.csv")
    vax = vax.rename(
        columns={"Location": "state", "Date": "date", "Administered": "admin"}
    )
    vax["date"] = pd.to_datetime(vax["date"])

    # start date
    start_date = vax["date"].min()
    # end date
    end_date = gun.index.max()

    vax = vax[(vax["date"] <= end_date) & (vax["date"] >= start_date)]

    vax = vax[["date", "state", "Administered_Dose1_Recip_65PlusPop_Pct"]]
    vax["Administered_Dose1_Recip_65PlusPop_Pct"] = (
        vax["Administered_Dose1_Recip_65PlusPop_Pct"] / 100.0
    )
    gun = gun[gun[(gun.index <= end_date) & (gun.index >= start_date)]][
        "fatal_incident_rate"
    ]
    print(ks_test2(vax["Administered_Dose1_Recip_65PlusPop_Pct"], gun))
    part_2()
