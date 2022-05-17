from datetime import datetime
import pandas as pd
from utilities import tukey, t_test, walds_test, z_test
from vax import *
import sys

# our states are Massachussetts and Mississippi.
GROUP_11_STATES = ["MA", "MS"]

if __name__ in "__main__":
    df = pd.read_csv("United_States_COVID-19_Cases_and_Deaths_by_State_over_Time.csv")
    df = df.rename(
        columns={"submission_date": "date", "tot_cases": "cases", "tot_death": "deaths"}
    )
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    dfs = {st: {} for st in GROUP_11_STATES}
    for state in GROUP_11_STATES:
        for col in ["cases", "deaths"]:
            print("Cleaning %s data in state %s" % (col, state))
            st = df[["date", "state", col]]
            st = st[st["state"] == state]
            st[col] = st[col].diff()
            stlen = st.shape[0]
            st.dropna(inplace=True)
            print(
                "Missing %d out of %d days of data" % (stlen - st.shape[0] - 1, stlen)
            )
            print("Applying Tukey's rule")
            stlen = st.shape[0]
            st = tukey(st, col)
            print("Number of rows with outliers: %d" % (stlen - st.shape[0]))
            dfs[state][col] = st
            #### PART A ####
            feb_21 = st[
                (st["date"] >= datetime(2021, 2, 1))
                & (st["date"] <= datetime(2021, 2, 28))
            ]
            mar_21 = st[
                (st["date"] >= datetime(2021, 3, 1))
                & (st["date"] <= datetime(2021, 3, 31))
            ]

            # get population means and standard deviations
            pop_mu = np.mean(df[col])
            pop_stdev = np.std(df[col])
            null_hypo = float(0)

            # run hypothesis tests
            for t in (t_test, walds_test, z_test):
                if walds_test:
                    null_hypo = pop_mu
                feb_test_stat, feb_p_val = t(
                    feb_21[col],
                    pop_mean=pop_mu,
                    **{"pop_stdev": pop_stdev, "null_hypo": null_hypo}
                )
                mar_test_stat, mar_p_val = t(
                    mar_21[col],
                    pop_mean=pop_mu,
                    **{"pop_stdev": pop_stdev, "null_hypo": null_hypo}
                )

        print()

    vax = pd.read_csv("COVID-19_Vaccinations_in_the_United_States_Jurisdiction.csv")
    vax = vax.rename(
        columns={"Location": "state", "Date": "date", "Administered": "admin"}
    )
    vax["date"] = pd.to_datetime(vax["date"])
    # select out just date, admin, and state field
    # sort by date
    vax = vax[["date", "state", "admin"]].sort_values("date", ascending=True)
    # select our states
    state_dfs = []
    for state in GROUP_11_STATES:
        st = vax[vax["state"] == state]
        # decumulatize
        st["admin"] = st["admin"].diff()
        # drop any missing data
        stlen = st.shape[0]
        st.dropna(inplace=True)
        print(
            "Missing %d days of vaccine data in %s" % (stlen - st.shape[0] - 1, state)
        )
        print("Applying Tukey's rule")
        stlen = st.shape[0]
        st = tukey(st, "admin")
        print("Rows with outliers in vaccine data: %d" % (stlen - st.shape[0]))
        state_dfs.append(st)
    print()

    ### PART D ###
    for state, name in zip(state_dfs, GROUP_11_STATES):
        part_d(state, name)

    ### PART E ###
    part_e(*state_dfs, *GROUP_11_STATES)
