# 5 ways of generalization classifier

# SVM
from sklearn import svm
model = svm.SVC(
kernel = linear, # or poly or rbf
degree = 3,    # used only under poly
gamma = 0.7,   # used only under rbf
probability = True, #predict class by prob
C = 0.5)       # smaller C get stronger regularization

model.fit(x,y)
model.predict_proba(x[i:i+1:1,:]) #svm prob output
model.predict(x[i:i+1:1,:]) # max prob as class

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(
	criterion='gini', 
	max_depth=None,
	min_samples_leaf=1,
    min_samples_split=2)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(
	n_estimators=50,	#number of trees
	bootstrap=True,		#bagging
	criterion='gini',
    max_depth=5,
    max_features='auto',	#sqrt(features)
	min_samples_leaf=1,
    min_samples_split=2)

# Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(
	n_estimators=50,	#number of trees
    learning_rate=0.1,
    max_depth=3,
    max_features='None')	#sqrt(features)

# XGB Classifier
from xgboost import XGBClassifier
clf = XGBClassifier(
	n_estimators=50,	#number of trees
	learning_rate=0.1,
	min_child_weight=1,  # sum of h of evey leaf
	max_depth=3,
	alpha= 0.9,     #l1 loss regularization, lasso
	lambda= 0.999,   #l2 loss regularization, ridge 
	scale_pos_weight = 1) 
	# use for inblance data, add weight of miner data class

from sklearn.metrics import accuracy_score
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(accuracy_score(y_test, y_pred))
print(clf.feature_importances_)