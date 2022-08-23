import pandas as pd
import csv


def parse_arguments():
	# Create argument parser
	parser = argparse.ArgumentParser()
	# Positional mandatory arguments
	parser.add_argument("data_path",
						help="Path where credits data is.",
						type=str)
	# Print version
	parser.add_argument("--version", action="version",
											version='%(prog)s - Version 1.0')

	# Parse arguments
	args = parser.parse_args()

	return args


def get_features_map(global_data_path, credits_path_dict):
	# Load global database
	global_df = pd.read_csv(global_data_path, nrows=1)

	# Load files databases
	dfs_dict = {}
	for name in credits_path_dict.items():
		dfs_dict[name[0]] = pd.read_csv(name[1], nrows=1)

	# Create Look Up Table between global database features
	# and files database features
	# Create Look Up Table to have correspondance
	#  between features in database and feature in original dataframes
	features_map = pd.DataFrame(index=global_df.columns, columns=[
		'TableName', 'Row'])
	for col in global_df.columns:
		for df in dfs_dict.items():
			for col2 in df[1].columns:
				if col2 in col:
					features_map.loc[col, 'TableName'] = df[0]
					features_map.loc[col, 'Row'] = col2

	return features_map


def get_unique_items(credits_path_list):
	unique_items_df = pd.DataFrame()
	for file in credits_path_list:
		dataframe = pd.read_csv(file)
		df_obj = dataframe.select_dtypes(include='object')
		for column in df_obj.columns:
			if column not in unique_items_df.columns:
				unique_items_df = pd.concat([unique_items_df, pd.DataFrame(
					df_obj[column][df_obj[column].notnull()].unique(),
					columns=[column])], axis=1)
			else:
				list1 = list(unique_items_df[column])
				list2 = list(df_obj[column].unique())
				list1.extend(list2)
				new_set = set(list1)
				unique_items_df.drop(column, inplace=True, axis=1)
				unique_items_df = pd.concat([unique_items_df, pd.DataFrame(
					df_obj[column].unique(), columns=[column])], axis=1)

	unique_items_df.loc[-1] = 'undefined'
	unique_items_df.index = unique_items_df.index + 1  # shifting index
	unique_items_df.sort_index(inplace=True)

	return unique_items_df


if __name__ == "__main__":
	args = parse_arguments()

	data_path = args.data_path
	global_data_path = data_path+'/global_train_data.csv'
	credits_path_dict = {
		'application_train': data_path+'/application_train.csv',
		'bureau': data_path+'/bureau.csv',
		'bureau_balance': data_path+'/bureau_balance.csv',
		'previous_application': data_path+'/previous_application.csv',
		'POS_CASH_balance': data_path+'/POS_CASH_balance.csv',
		'installments_payments': data_path+'/installments_payments.csv',
		'credit_card_balance': data_path+'/credit_card_balance.csv'
	}
	features_map = get_features_map(global_data_path, credits_path_dict)
	# Save LUT
	features_map.to_csv(data_path+'features_map.csv', index=[0])

	credits_path_list = [
		data_path+'application_train.csv',
		data_path+'/credit_card_balance.csv',
		data_path+'/bureau_balance.csv',
		data_path+'/POS_CASH_balance.csv',
		data_path+'/bureau.csv',
		data_path+'/installments_payments.csv',
		data_path+'/previous_application.csv'
	]
	unique_items_df = get_unique_items(credits_path_list)
	unique_items_df.to_csv(data_path+'unique_items_df.csv', index=False)
