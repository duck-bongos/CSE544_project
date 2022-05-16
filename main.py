from datetime import datetime
import pandas as pd
from utilities import tukey
from vax import *

# our states are Massachussetts and Mississippi.
GROUP_11_STATES = ["MA", "MS"]

if __name__ in "__main__":
	vax = pd.read_csv("COVID-19_Vaccinations_in_the_United_States_Jurisdiction.csv")
	vax = vax.rename(columns={"Location": "state", "Date": "date", 'Administered': 'admin'})
	vax["date"] = pd.to_datetime(vax["date"])
	# select out just date, admin, and state field
	# sort by date
	vax = vax[['date', 'state', 'admin']].sort_values("date", ascending=True)
	# select our states
	state_dfs = []
	for state in GROUP_11_STATES:
		st = vax[vax['state'] == state]
		# decumulatize
		st['admin'] = st['admin'].diff()
		# drop any missing data
		stlen = st.shape[0]
		st.dropna(inplace=True)
		print('Missing %d days of vaccine data in %s' % (stlen - st.shape[0] - 1, state))
		print("Applying Tukey's rule")
		stlen = st.shape[0]
		st = tukey(st, 'admin')
		print("Rows with outliers in vaccine data: %d" % (stlen - st.shape[0]))
		state_dfs.append(st)
	print()

	### PART D ###
	for state, name in zip(state_dfs, GROUP_11_STATES):
		part_d(state, name)
	
	### PART E ###
	part_e(*state_dfs, *GROUP_11_STATES)

