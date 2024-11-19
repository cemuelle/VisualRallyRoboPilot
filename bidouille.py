import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
from pandas.core.common import flatten
import lzma
import pickle
from sklearn.preprocessing import PolynomialFeatures

# reading dataset
speed = []
forward = []
backward = []
with lzma.open("record_red_0.npz", "rb") as file:
	data = pickle.load(file)
	for i in data:
		speed.append(i.car_speed)
		forward.append(i.current_controls[0])
		backward.append(i.current_controls[1])
with lzma.open("record_cyan_0.npz", "rb") as file:
	data = pickle.load(file)
	for i in data:
		speed.append(i.car_speed)
		forward.append(i.current_controls[0])
		backward.append(i.current_controls[1])
with lzma.open("record_blue_0.npz", "rb") as file:
	data = pickle.load(file)
	for i in data:
		speed.append(i.car_speed)
		forward.append(i.current_controls[0])
		backward.append(i.current_controls[1])

# transforming dataset
dataset_raw = pd.DataFrame({'speed': speed, 'forward': forward, 'backward': backward})
dataset_raw["speed_t-1"] = dataset_raw["speed"].shift(1)
dataset_raw = dataset_raw[1:]


# splitting dataset
X = dataset_raw[["speed_t-1","forward","backward"]]
y = dataset_raw[["speed"]]

# poly
poly_features = PolynomialFeatures(degree=2) # decide the maximal degree of the polynomial feature
X_ploy = poly_features.fit_transform(X)

# reg fit
reg = LinearRegression().fit(X_ploy[1:], y[1:])
#reg = LinearRegression().fit(X[1:], y[1:])

# predictions
tab_preds = [0]
for ind in X.index:
	newX = np.array([float(tab_preds[len(tab_preds)-1]),float(X['forward'][ind]),float(X['backward'][ind])]).reshape(1, -1)
	newX_ploy = poly_features.fit_transform(newX)
	pr = reg.predict(newX_ploy)
	#pr = reg.predict(newX)
	'''
	pr = 0
	if X['forward'][ind] == 1:
		pr = tab_preds[len(tab_preds)-1] + 7
	elif X['backward'][ind] == 1:
		pr = tab_preds[len(tab_preds)-1] - 7
	else:
		pr = tab_preds[len(tab_preds)-1] - 3
	'''
	if pr > 50:
		pr = 50
	if pr < -15:
		pr = -15
	
	tab_preds.append(pr)


tab_preds = list(flatten(tab_preds))

mae = mean_absolute_error(y["speed"], tab_preds[:-1])
print(mae)


plt.plot(y["speed"], color = 'g')
plt.plot(tab_preds[:-1], color = 'r')
plt.show()