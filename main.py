from datetime import datetime
import pandas as pd
from utilities import (
	tukey,
	t_test,
	two_sample_t_test,
	two_sample_walds_test,
	walds_test,
	z_test,
)
from vax import *
from cases_deaths import *
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

	oldstdout = sys.stdout
	sys.stdout = open('cleaning.log', 'w')

	out_to_file = []
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
	sys.stdout.close()

	sys.stdout = open('part_a.log', 'w')
	### PART A ###
	for state in dfs:
		for col in dfs[state]:
			part_a(dfs[state][col], state, col)
	sys.stdout.close()
	
	### PART B ###
	sys.stdout = open('part_b.log', 'w')
	part_b(dfs)
	sys.stdout.close()

	### PART C ###
	sys.stdout = open('part_c.log', 'w')
	part_c(dfs)
	sys.stdout.close()

	sys.stdout = open('cleaning.log', 'a')
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
	sys.stdout.close()

	### PART D ###
	sys.stdout = open('part_d.log', 'w')
	for state, name in zip(state_dfs, GROUP_11_STATES):
		part_d(state, name)
	sys.stdout.close()

	### PART E ###
	sys.stdout = open('part_e.log', 'w')
	part_e(*state_dfs, *GROUP_11_STATES)
	sys.stdout.close()
