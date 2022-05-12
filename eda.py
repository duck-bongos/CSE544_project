from datetime import datetime

import pandas as pd

if __name__ in "__main__":
    df = pd.read_csv("United_States_COVID-19_Cases_and_Deaths_by_State_over_Time.csv")
    df["submission_date"] = df["submission_date"].apply(
        lambda x: datetime.strptime(x, "%m/%d/%Y")
    )
    print(df.head())

    
