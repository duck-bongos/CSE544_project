from datetime import datetime

import pandas as pd


def de_cumulate_column():
    pass


# our states are Massachussetts and Mississippi.
GROUP_11_STATES = ["MA", "MS"]


if __name__ in "__main__":
    df = pd.read_csv("United_States_COVID-19_Cases_and_Deaths_by_State_over_Time.csv")
    vax = pd.read_csv("COVID-19_Vaccinations_in_the_United_States_Jurisdiction.csv")

    print(df.shape)
    print(vax.shape)

    # sort by date
    df["submission_date"] = pd.to_datetime(df["submission_date"])
    vax["Date"] = pd.to_datetime(vax["Date"])
    vax = vax.sort_values("Date", ascending=True)
    df = df.sort_values("submission_date", ascending=True)

    # filter to only the states I need and take the daily difference of total case increases
    df_ms = df[df["state"] == "MS"]
    df_ms[["tot_cases", "conf_cases", "prob_cases"]] = df_ms[
        ["tot_cases", "conf_cases", "prob_cases"]
    ].diff()
    df_ma = df[df["state"] == "MA"]
    df_ma[["tot_cases", "conf_cases", "prob_cases"]] = df_ma[
        ["tot_cases", "conf_cases", "prob_cases"]
    ].diff()

    vax_ms = vax[vax["Location"] == "MS"]
    vax_ma = vax[vax["Location"] == "MA"]

    # what columns do I have? what columns need to be .diff() 'ed?

    # .apply(
    #    lambda x: datetime.strptime(x, "%m/%d/%Y")
    # )
    print(df.head())
