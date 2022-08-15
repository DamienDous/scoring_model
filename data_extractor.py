import argparse
import time
from contextlib import contextmanager

import pandas as pd
import numpy as np

from data_generation import data_generation


@contextmanager
def timer(title):
	t0 = time.time()
	yield
	print("{} - done in {:.0f}s".format(title, time.time() - t0))


def parse_arguments():
	# Create argument parser
	parser = argparse.ArgumentParser()
	# Positional mandatory arguments
	parser.add_argument("cust_nb",
						help="Customer number to select.",
						type=int)
	# Optional arguments
	parser.add_argument(
		"-p", "--proc_db",
		help="Process database (True if not specified).",
		action='store_true')
	# Print version
	parser.add_argument("--version", action="version",
						version='%(prog)s - Version 1.0')

	# Parse arguments
	args = parser.parse_args()

	return args


# Dict containing base names of database files
tables_names = [
	'application_train',
	'bureau',
	'previous_application',
	'POS_CASH_balance',
	'installments_payments',
	'credit_card_balance',
]


def main(custumer_number, process_database=True):

	with timer("Load tables"):
		tables_dfs = {}
		for table_name in tables_names:
			# Read table df and save it in dict
			tables_dfs[table_name] = pd.read_csv('../data/'+table_name+'.csv')
		# Read bureau_balance table (not same way to get sample of it)
		bb = pd.read_csv('../data/bureau_balance.csv')

	# Create ramdom 'custumer_number' custumers idx list
	customer_idx_list = \
		tables_dfs['application_train'].sample(
			custumer_number)['SK_ID_CURR'].to_list()

	with timer("Get small tables"):
		sample_tables_dfs = {}
		for table_name, table_df in tables_dfs.items():
			sample_tables_dfs[table_name] = table_df[table_df['SK_ID_CURR'].isin(
				customer_idx_list)]

			if table_name == 'bureau':
				sample_tables_dfs['bureau_balance'] = \
					bb.loc[bb['SK_ID_BUREAU'].isin(
						sample_tables_dfs[table_name]['SK_ID_BUREAU'])]

	with timer("Write small tables"):
		for table_name, table_df in sample_tables_dfs.items():
			table_df.to_csv('data/'+table_name+'.csv')

	with timer("Process database"):
		if process_database:
			database = data_generation().process_database(sample_tables_dfs)

			global_df = pd.read_csv('../data/global_train_data.csv', nrows=1)

			# Fill missing data for missing columns
			missing_columns = list(set(global_df.columns) -
								   set(database.columns))
			database.loc[:, missing_columns] = np.nan
			database = database[global_df.columns]

			database.to_csv("data/global_small_data.csv")


if __name__ == "__main__":
	args = parse_arguments()

	with timer("Full extractor run"):
		main(args.cust_nb, args.proc_db)
