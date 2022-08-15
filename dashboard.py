import sys
import os
import warnings
import time
import io
import requests

from contextlib import contextmanager

import pandas as pd
import joblib

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import shap

import streamlit as st

from box_displayer import box_displayer
from data_generation import data_generation

st.set_option('deprecation.showPyplotGlobalUse', False)


tables_path = 'tables'
data_path = 'data'

if not os.path.isdir(tables_path):
	os.makedirs(tables_path)

if not os.path.isdir(data_path):
	os.makedirs(data_path)

url_pos_LUT = {
	'application_train':
	"https://www.dropbox.com/s/owyjir51ceoxyt0/\
			application_train.csv?dl=1",
	'bureau_balance':
	"https://www.dropbox.com/s/0wkjmwdw2dz96hl/\
			bureau_balance.csv?dl=1",
	'bureau':
	"https://www.dropbox.com/s/1s251hdtepkh2vt/\
			bureau.csv?dl=1",
	'credit_card_balance':
	"https://www.dropbox.com/s/7lgxgul6z17ti9f/\
			credit_card_balance.csv?dl=1",
	'installments_payments':
	"https://www.dropbox.com/s/plvagomvb0v26sk/\
			installments_payments.csv?dl=1",
	'POS_CASH_balance':
	"https://www.dropbox.com/s/o566ek9dheozv18/\
			POS_CASH_balance.csv?dl=1",
	'previous_application':
	"https://www.dropbox.com/s/iz8ibayau8gd5zl/\
			previous_application.csv?dl=1",
}


@contextmanager
def timer(title):
	t0 = time.time()
	yield
	print("{} - done in {:.0f}s".format(title, time.time() - t0))


# Use the full page instead of a narrow central column
st.set_page_config(layout="wide")


def get_dropbox_file(url):
	s = requests.get(url).content
	return io.StringIO(s.decode('utf-8'))


@st.cache
def get_unique_items_dict():  # Get unique items dict as static
	path = tables_path+'/unique_items_df.csv'
	if os.path.exists(path):
		print(path, 'exists')
		df = pd.read_csv(path)
	else:
		print(path, 'does not exist')
		url = "https://www.dropbox.com/s/wuxenueynwjqlud/\
			unique_items_df.csv?dl=1"
		df = pd.read_csv(get_dropbox_file(url))
		df.to_csv(path)
	return {column: list(
		df[column][df[column].notnull()]) for column in df.columns}


@st.cache
def get_columns_description_df():  # Get description df as static
	path = tables_path+'/HomeCredit_columns_description_improved.csv'
	if os.path.exists(path):
		print(path, 'exists')
		columns_description = pd.read_csv(path, sep=',')
	else:
		print(path, 'does not exist')
		url = "https://www.dropbox.com/s/3r6yrtc3cm7h9b3/\
			HomeCredit_columns_description_improved.csv?dl=1"
		columns_description = pd.read_csv(get_dropbox_file(url), sep=';')
		columns_description.to_csv(path)
	return columns_description


@st.cache
def get_features_lut():  # Get unique items dict as static
	path = tables_path+'/features_lut.csv'
	if os.path.exists(path):
		print(path, 'exists')
		features_lut = pd.read_csv(path, index_col=[0])
	else:
		print(path, 'does not exist')
		url = "https://www.dropbox.com/s/4g1dvziylqwbmku/features_lut.csv?dl=1"
		features_lut = pd.read_csv(
			get_dropbox_file(url), index_col=[0])
		features_lut.to_csv(path)
	return features_lut


@st.cache
def get_global_df():  # Load database_columns as static
	path = data_path+'/global_small_data.csv'
	if os.path.exists(path):
		print(path, 'exists')
		global_df = pd.read_csv(path, index_col=[0])
	else:
		print(path, 'does not exist')
		url = "https://www.dropbox.com/s/qgav4sf8p3hmz2n/\
			global_small_data.csv?dl=1"
		global_df = pd.read_csv(get_dropbox_file(url), index_col=[0])
		global_df.to_csv(path)
	return global_df


@st.cache
def get_credit_dfs():
	credits_dfs = {}
	for file, key in url_pos_LUT.items():
		path = data_path+'/'+file+'.csv'
		if os.path.exists(path):
			print(path, 'exists')
			credits_dfs[file] = pd.read_csv(path, index_col=[0])
		else:
			print(path, 'does not exist')
			credits_dfs[file] = pd.read_csv(
				get_dropbox_file(key), index_col=[0])
			credits_dfs[file].to_csv(path)
	return credits_dfs


@st.cache
def get_pipeline():  # Load pipeline
	path = data_path+'/pipeline.joblib'
	if not os.path.exists(path):
		print(path, 'does not exist')
		url = "https://www.dropbox.com/s/1btc9mrqy4hdsd1/pipeline.joblib?dl=1"
		s = requests.get(url).content
		with open(data_path+'/pipeline.joblib', 'wb') as f:
			f.write(s)
		del s
	pipeline = joblib.load(data_path+'/pipeline.joblib')
	return pipeline


@st.cache
def get_shap_explainer():
	path = data_path+'/explainer.joblib'
	if not os.path.exists(path):
		print(path, 'does not exist')
		url = "https://www.dropbox.com/s/ano5bdbnszohk2x/explainer.pkl?dl=1"
		s = requests.get(url).content
		with open(data_path+'/explainer.joblib', 'wb') as f:
			f.write(s)
		del s
	shap_explainer = joblib.load(data_path+'/explainer.joblib')
	return shap_explainer


@st.cache
def get_proba_prediction(clf, scaled_df):
	proba_prediction = clf.predict_proba(scaled_df)
	proba_prediction_df = pd.DataFrame(proba_prediction, index=scaled_df.index)
	return proba_prediction_df


@st.cache
def transform_global_df(pipeline, df):
	# Get pipeline SimpleImputer and StandardScaler
	imputer = pipeline[0]
	scaler = pipeline[1]
	impute_data = imputer.transform(df)
	scaled_data = scaler.transform(impute_data)
	scaled_data_df = pd.DataFrame(
		scaled_data, columns=df.columns, index=df.index)
	return scaled_data_df


@st.cache
def get_tree_explainer(clf, scaled_df):
	explainer = shap.TreeExplainer(clf, scaled_df)
	return explainer


with timer("Instantiation process"):
	# Get unique items dict as static
	unique_items_dict = get_unique_items_dict()
	# Get features LUT
	features_lut = get_features_lut()
	# Get columns description df as static
	cols_des_df = get_columns_description_df()
	# Load model as static
	pipeline = get_pipeline()
	# if credit dataframes does not exist, load it and write it
	credits_dfs = get_credit_dfs()
	# Load database_columns as static
	global_df = get_global_df()
	global_df = global_df.set_index('SK_ID_CURR')
	# Get Target column as df
	target_df = pd.DataFrame(global_df['TARGET'],
							 index=global_df.index,
							 columns=['TARGET'])
	# Get X and y value from data
	global_df.drop(['TARGET'], axis=1, inplace=True)
	# Create customer idx list
	customer_idx_list = global_df.index.to_list()
	# Transform X to have data for input of classifier
	scaled_data_df = transform_global_df(pipeline, global_df)
	loan_proba_df = get_proba_prediction(pipeline[3], scaled_data_df)
	# SHAP explainer values (NumPy array)
	tree_explainer = get_tree_explainer(pipeline[3], scaled_data_df)
	shap_explainer = get_shap_explainer()


def plot_histo_with_hline(values0, values1, hline_pos):
	fig, ax = plt.subplots(figsize=(10, 3))
	y, x, _ = ax.hist([values0, values1], bins=200,
					  label=['accepted', 'rejected'])
	ax.axvline(hline_pos, color='red', linewidth=5)
	return fig


def request_prediction(model_uri, data):
	headers = {"Content-Type": "text/csv"}

	data_csv = data.to_csv()
	response = requests.request(
		method='POST', headers=headers, url=model_uri, data=data_csv)
	try:
		assert (response.status_code == 200), (
			"Request failed with status {}, {}".format(
				response.status_code, response.text))
	except Exception as e:
		print(e)
		warnings.warn('WARNING : Prediction is done from local predictor')
		return pipeline.predict_proba(data)[0]
	else:
		return response.json()['prediction']


row0 = st.columns(1)

# Display dashboard header
row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns(
	(.1, 2, .1, 1, .1))

row0_1.title('Home Credit Default Risk')

with row0_2:
	st.write('')

row0_2.subheader(
	'A Streamlit web app by [Damien Dous](damien.dous@gmail.com)')

row1_spacer1, row1_1, row1_spacer2 = st.columns((.1, 3.2, .1))

with row1_1:
	st.markdown(
		"**To begin, please enter current customer idx.** 👇")

# Display customer id selection panel
row2_spacer1, row2_1, row2_spacer2 = st.columns((.1, 3.2, .1))
no_idx_curr = True
with row2_1:
	select_idx_curr = st.selectbox(
		"Select one of our sample customer profiles idx",
		customer_idx_list[:10])
	st.markdown("**or**")
	text = st.empty()
	text_idx_curr = text.text_input(
		"Input your customer idx that you want \
		to predict if loan is accepted or declined")
	if text_idx_curr != '' and text_idx_curr.isdigit() \
			and int(text_idx_curr) in customer_idx_list:
		idx_curr = int(text_idx_curr)
	else:
		idx_curr = int(select_idx_curr)

	if text_idx_curr != '' and text_idx_curr.isdigit() and \
			int(text_idx_curr) not in customer_idx_list:
		st.markdown("custumer idx does not exist, it will display " +
					str(idx_curr)+" custumer idx")

# Display predict button
select_btn = st.button('Select')

# When predict button is selected:
if ('idx_curr' in st.session_state and
		st.session_state['idx_curr'] == idx_curr) or select_btn:

	st.session_state['idx_curr'] = idx_curr

	idx_curr_pos = customer_idx_list.index(idx_curr)

	# Separate dashboard in two columns
	row3_space1, row3_1, row3_space2, row3_2, row3_space3 = st.columns(
		(.1, 1, .1, 1, .1))
	with row3_1:
		st.subheader('Values that can be modifiable')

	with row3_2:
		st.subheader('Values that can not be modifiable')

	# Load data of the current id
	id_credit_dfs = {}
	for df_name, credits_df in credits_dfs.items():
		if df_name == 'bureau_balance':
			continue

		id_credit_dfs[df_name] = credits_df[credits_df['SK_ID_CURR'] ==
											idx_curr].reset_index(drop=True)

		if df_name == 'bureau':
			df = credits_dfs['bureau_balance']
			id_credit_dfs['bureau_balance'] = df.loc[
				df['SK_ID_BUREAU'].isin(
					id_credit_dfs[df_name]['SK_ID_BUREAU'])]\
				.reset_index(drop=True)

	# Copy dataframes dict to not modified them directly
	id_credit_dfs_ = {key: id_credit_dfs[key].copy()
					  for key in id_credit_dfs}
	# Create SHAP explainer dataframe for current data with initial data
	shap_values = tree_explainer.shap_values(scaled_data_df.loc[idx_curr, :],
											 check_additivity=False)
	shap_values_df = pd.DataFrame(shap_values,
								  index=scaled_data_df.columns,
								  columns=['Value'])
	shap_values_df['Percentage'] = shap_values_df['Value'].abs() / \
		shap_values_df['Value'].abs().sum()

	# Sort SHAP dataframe
	shap_sorted_features = list(
		shap_values_df['Value'].abs().sort_values(ascending=False).index)
	shap_sorted_values_df = shap_values_df.loc[shap_sorted_features]

	st.write('')

	editable_counter = 0
	counter = 0
	while editable_counter < 10:
		db_feature = shap_sorted_values_df.index[counter]
		# Check if feature is in features LUT
		if db_feature not in features_lut.index:
			print('ERROR : '+db_feature+' feature not in index')
			counter += 1
			continue
		feature_info = features_lut.loc[db_feature, :]
		# Check if feature has correspondance in LUT
		if pd.isnull(feature_info['TableName']):
			print('WARNING : '+db_feature+' not refind in LUT')
			counter += 1
			continue
		# Do not display ***_balance file value
		# because it contains more than one line for a customer
		if feature_info['TableName'] in ['POS_CASH_balance',
										 'credit_card_balance',
										 'bureau_balance']:
			print('WARNING : '+db_feature+' feature in balance tables')
			counter += 1
			continue

		# Get column description row for the considered feature
		column_description = cols_des_df.loc[(
			cols_des_df['Row'] == feature_info['Row']) & (
			cols_des_df['Table'] == feature_info['TableName']+'.csv'), :]
		# Process description for the considered feature
		description_text = column_description['Description Small'].iloc[0]\
			+ ' - SHAP coef :' + \
			'{:.3f}'.format(shap_sorted_values_df.loc[db_feature, 'Value'])
		# Get type for the considered feature
		type_ = column_description['Type'].iloc[0]

		# Dispatch feature according it's considered fixed or editable
		if column_description['State'].iloc[0] == 'EDITABLE':
			# Display fixed feature on left
			bd = box_displayer(unique_items_dict, row3_1)
			bd.display_editable_feature(
				id_credit_dfs_[feature_info['TableName']],
				feature_info['Row'],
				description_text,
				type_)
			editable_counter += 1
		elif column_description['State'].iloc[0] == 'FIXED':
			# Display fixed feature on left
			bd = box_displayer(unique_items_dict, row3_2)
			bd.display_fixed_feature(
				id_credit_dfs_[feature_info['TableName']],
				feature_info['Row'],
				description_text,
				type_)
		counter += 1

	# Add predict button
	predict_btn = st.button('Prédire')

	# When predict button is activate
	if predict_btn or ('predict' in st.session_state
					   and st.session_state['predict'] == idx_curr):
		st.session_state['predict'] = idx_curr

		# DATA GENERATION FOR CLASSIFIER
		# Create data_genration object
		dg = data_generation()
		# Process data from current customer database
		customer_data = dg.process_database(id_credit_dfs_)
		customer_data.drop(['TARGET', 'SK_ID_CURR'], axis=1, inplace=True)
		# Fill missing data for missing columns
		missing_columns = list(set(global_df.columns) -
							   set(customer_data.columns))
		customer_data.loc[0, missing_columns] = global_df.loc[
			global_df.index == idx_curr, missing_columns].values[0]
		customer_data = customer_data[global_df.columns]

		# PREDICTION FROM MODEL
		# make a prediction request to flask server
		FLASK_URI = 'https://scoringmodelapi.herokuapp.com/api/predict'
		# CHECK SERVER IS RUNNING AND CHECK SERVER RETURN GOOD VALUES
		server_pred = request_prediction(FLASK_URI, customer_data)

		col4_1, col4_2, col4_3 = st.columns((1, .2, 1))
		with col4_1:
			st.metric('Custumer client number',
					  idx_curr)

		row4_1, row4_2, row4_3 = st.columns((.7, 1, .7))
		with row4_2:
			# Plot gauge with result
			fig = go.Figure(go.Indicator(
				domain={'x': [0, 1], 'y': [0, 1]},
				value=server_pred[0],
				mode="gauge+number+delta",
				title={'text': "Score"},
				delta={'reference': 0.5},
				gauge={'axis': {'range': [None, 1]},
					   'steps': [
					{'range': [0, 0.5], 'color': "lightgrey"},
					{'range': [0.5, 1], 'color': "lightgreen"}
				],
					'threshold':
					{'line':
					 {'color': "red", 'width': 8},
					 'thickness': 1,
					 'value': 0.5
					 }
				}))
			st.plotly_chart(fig)

		row5_1, row5_2, row5_3, row5_4, row5_5 = st.columns(
			(.1, 1, .1, 1, .1))
		with row5_2:
			# Display the most important features with coefficient
			# plot the SHAP values for the output of the current idx data
			# Create SHAP explainer dataframe for current data with initial
			# data
			st.subheader('SHAP values for this customer')
			max_display = 10
			# use SHAP explainer to explain test set predictions
			waterfall_plot = shap.plots.waterfall(shap_explainer[idx_curr_pos],
												  max_display=max_display)
			st.pyplot(waterfall_plot)
		with row5_4:
			st.subheader('Classifier features importance')
			feature_imp = pd.DataFrame(sorted(zip(
				pipeline[3].feature_importances_, customer_data.columns)),
				columns=['Value', 'Feature'])

			fig, ax = plt.subplots(figsize=(10, 6.5))
			sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(
				by="Value", ascending=False)[:10])
			plt.title('LightGBM Features (avg over folds)')
			st.pyplot(fig)

		row6_1, row6_2, row6_3 = st.columns((.7, 1, .7))
		with row6_2:
			accepted_loan_proba = loan_proba_df.loc[
				target_df["TARGET"] == 0, 0]
			declined_loan_proba = loan_proba_df.loc[
				target_df["TARGET"] == 1, 0]

			st.subheader('Custumer score compared to other customers')
			st.pyplot(plot_histo_with_hline(accepted_loan_proba,
											declined_loan_proba,
											server_pred[0]))

		row7_1, row7_2, row7_3, row7_4, row7_5 = st.columns(
			(.1, 1, .1, 1, .1))

		with row7_2:
			feature_list1 = feature_imp.sort_values(
				by="Value", ascending=False).loc[:10, 'Feature']
			feature1 = st.selectbox(
				"Select first feature to see distribution",
				feature_list1)

		with row7_4:
			feature_list2 = feature_imp.sort_values(
				by="Value", ascending=False).loc[:10, 'Feature']
			feature2 = st.selectbox(
				"Select second feature to see distribution",
				feature_list2, index=1)

		# Add visualization button
		visualization_btn = st.button('Visualize')
		# When visualization button is activate
		if visualization_btn or ('visualize' in st.session_state and
								 st.session_state['visualize'] == idx_curr):
			st.session_state['visualize'] = idx_curr
			row8_1, row8_2, row8_3, row8_4, row8_5 = st.columns(
				(.1, 1, .1, 1, .1))
			with row8_2:
				accepted_feature1_values = global_df.loc[target_df["TARGET"]
														 == 0, feature1]
				declined_feature1_values = global_df.loc[target_df["TARGET"]
														 == 1, feature1]
				st.subheader('Customer position for '+feature1)
				st.pyplot(plot_histo_with_hline(
					accepted_feature1_values.dropna(),
					declined_feature1_values.dropna(),
					global_df.loc[idx_curr, feature1]))

			with row8_4:
				accepted_feature2_values = global_df.loc[target_df["TARGET"]
														 == 0, feature2]
				declined_feature2_values = global_df.loc[target_df["TARGET"]
														 == 1, feature2]
				st.subheader('Customer position for '+feature2)
				st.pyplot(plot_histo_with_hline(
					accepted_feature2_values.dropna(),
					declined_feature2_values.dropna(),
					global_df.loc[idx_curr, feature2]))

			row9_1, row9_2, row9_3 = st.columns((1, 1, 1))
			with row9_2:
				st.subheader(feature2+' VS ' + feature2 +
							 ' with customer position')
				fig, ax = plt.subplots(figsize=(10, 10))
				sns.scatterplot(x=global_df[feature1],
								y=global_df[feature2],
								hue=loan_proba_df.loc[:, 0])
				ax.scatter(global_df.loc[idx_curr, feature1],
						   global_df.loc[idx_curr, feature2],
						   s=500,
						   color="red")
				st.pyplot(fig)
