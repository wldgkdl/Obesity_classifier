
from flask import Flask, render_template, request, flash, redirect, url_for

from joblib import dump, load
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
import pickle

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from tensorflow.keras.models import load_model

import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# 1. Decision Tree 
saved_clf = load('trained_models/clf_after_grid.joblib')

# 2. Random Forest
saved_rf = load('trained_models/rf_after_grid.joblib')

# 3. Logistic Regression
saved_LR = pickle.load(open('trained_models/saved_LR.sav', 'rb'))

# 4. Support Vector Machine 
saved_svc = load('trained_models/svc.joblib')

# 5. Naive Bayes
saved_NB = load('trained_models/nb.joblib')

# 6. Neural Network
saved_NN = load_model('trained_models/trained_NN.h5')



app = Flask(__name__)
app.secret_key = "wjdghks3#"


@app.route("/spec", methods = ["POST", "GET"])
def spec():
	return render_template("question_form.html")

@app.route("/form", methods = ["POST"])
def form():

	# Bring personal spec from the form
	height = request.form.get('height input')
	weight = request.form.get('weight input')
	gender = request.form.get('gender')
	h_unit = request.form.get('Height unit')
	w_unit = request.form.get('Weight unit')
	original_h = height
	original_w = weight
	original_g = gender
	w_unit = str("unit: ") + w_unit.capitalize()
	h_unit = str("unit: ") + h_unit.capitalize()

	# Modify spec based on unit
	if w_unit == 'lb':
		weight = int(0.453592 * float(weight))

	if h_unit == 'inches':
		height = int(2.54 * float(height))
	elif h_unit == 'feet':
		height = int(30.48 * float(height))

	if gender == 'Female':
		gender = 1
	else:
		gender = 0

	# Start inference	
	data = [gender, height, weight]

	DT = saved_clf.predict([data])
	DT_proba = saved_clf.predict_proba([data])
	print("The results of Decision Tree is",DT[0])

	RF = saved_rf.predict([data])
	print("The results of Random Forest is",RF[0])

	SVC = saved_svc.predict([data])
	print("The results of SVC is",SVC[0])

	NB = saved_NB.predict([data])
	print("The results of Naive Bayes is",NB[0])

	my_scaler = joblib.load('trained_models/scaler.gz')
	transformed_data = data[:]

	LR = saved_LR.predict(my_scaler.transform([transformed_data]))
	print("The results of Logistic Regression is",LR[0])

	NN = saved_NN.predict(my_scaler.transform([transformed_data]))
	print("The results of Neural Network is",np.argmax(NN))

	Ensemble_hard_voted = (DT + RF + SVC + NB + LR + np.argmax(NN))/6
	print("The results of Ensemble model is",int(round(Ensemble_hard_voted[0])))

	def index2class(result):
		index_info = ['Extremly weak', 'Weak', 'Normal', 'Overweight','Obesity','Extreme obesity']
		final_class = index_info[result[0]]
		return final_class

	def return_proba(result):
		int_proba = int(max(result[0])*100)
		return int_proba

	def return_color(result):
		color_info = ['Olive', 'DarkGreen', 'LimeGreen', 'Orange', 'OrangeRed', 'Red']
		final_color = color_info[result[0]]
		# ans = '{ "fill": ["color", "#eeeeee"], "innerRadius": 70, "radius": 85 }'
		# final_color = ans.replace('color', final_color)
		return final_color

	DT_class = index2class(DT)
	DT_proba = return_proba(DT_proba)
	DT_color = return_color(DT)




	
	return render_template("index2.html", original_h = original_h,
										  original_w = original_w,
										  original_g = original_g,
										  h_unit = h_unit,
										  w_unit = w_unit,
										  DT_class = DT_class,
										  DT_proba = DT_proba,
										  DT_color = DT_color)


if __name__ == '__main__':
	os.environ['FLASK_ENV'] = 'development'
	app.run(debug = True)





   

