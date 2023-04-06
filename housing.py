import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

print("Reading data...")
housing = pd.read_csv("datasets/housing.csv")
print(housing.head())
print(housing.describe())
print(housing.info())
print(housing["ocean_proximity"].value_counts())
print("_______________________________________\n")

print("Plotting histograms...")
housing.hist(bins=50, figsize=(20,15))
plt.savefig("housing_hist.png", bbox_inches='tight')
print("_______________________________________\n")

print("Splitting data to training and test set...")
bins = np.linspace(0, max(housing["median_house_value"]), 50)
# Save discretized Y values in a new array, broken down by the bins created above.
mhv = np.digitize(housing["median_house_value"], bins)
# this is needed to make stratified train/test sets
L, T = train_test_split(housing, test_size=0.2, random_state=42, stratify=mhv)
print("Learning set size: {:d}\nTest set size: {:d}".format(len(L), len(T)))
print("_______________________________________\n")


#housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
#housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
#split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

#for train_index, test_index in split.split(housing, mhv):
    # L ... learning/training set
    #L = housing.loc[train_index]
    # T ... test set
    #T = housing.loc[test_index]


L.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
plt.savefig("housing_lat_lon.png", bbox_inches='tight')

L.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
       s=L["population"]/100, label="population",
       c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
plt.legend()

plt.savefig("housing_population.png", bbox_inches='tight')
#plt.show()


from pandas.plotting import scatter_matrix
print("Computing correlations...")
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(L[attributes], figsize=(12, 8))
plt.savefig("housing_correlations.png", bbox_inches='tight')
print("_______________________________________\n")

#L["rooms_per_household"] = L["total_rooms"] / L["households"]
#L["bedrooms_per_room"] = L["total_bedrooms"] / L["total_rooms"]
#L["population_per_household"] = L["population"] / L["households"]

print("Attribute correlations with class....")
C = L.corr()
print(C["median_house_value"].sort_values(ascending=False))
print("_______________________________________\n")


#Prepare the data for Machine Learning algorithms
housing = L.drop("median_house_value", axis=1) # drop labels for training set
housing_labels = L["median_house_value"].copy()

#housing.dropna(subset=["total_bedrooms"]) # option 1
#housing.drop("total_bedrooms", axis=1) # option 2
# option 3
#median = L["total_bedrooms"].median()
#L["total_bedrooms"].fillna(median, inplace=True)

#sample_incomplete_rows = L[L.isnull().any(axis=1)].head()
#print(sample_incomplete_rows)

#print("Handling Categorical Attributes....")
#housing_cat = L["ocean_proximity"]

#from sklearn.preprocessing import LabelBinarizer
#encoder = LabelBinarizer(sparse_output=True)
#housing_cat_1hot = encoder.fit_transform(housing_cat)
#print("_______________________________________\n")

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer

rooms_ix, bedrooms_ix, population_ix, household_ix = [list(housing.columns).index(col) for col in ("total_rooms", "total_bedrooms", "population", "households")]

def add_extra_features(X):
    rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
    population_per_household = X[:, population_ix] / X[:, household_ix]
    return np.c_[X, rooms_per_household, population_per_household]

attr_adder = FunctionTransformer(add_extra_features, validate=False)

all_attributes = list(housing.columns)+["rooms_per_household", "population_per_household"]
housing_extra_attribs = attr_adder.fit_transform(housing.values)
housing_extra_attribs = pd.DataFrame(
    housing_extra_attribs,
    columns = all_attributes)
#housing_extra_attribs.head()

housing_num = housing.drop("ocean_proximity", axis=1)
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

from sklearn.compose import ColumnTransformer

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', FunctionTransformer(add_extra_features, validate=False)),
        ('std_scaler', StandardScaler()),
    ])
housing_num_tr = num_pipeline.fit_transform(housing_num)

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])
print("Finishing data preparation...")
housing_prepared = full_pipeline.fit_transform(housing)
print("{}".format(housing_prepared.shape))
print("_______________________________________\n")

# Correlations with class after the pipeline
#for i in range(housing_prepared.shape[1]):
#    print("{:6.2f}".format(np.corrcoef(housing_prepared[:,i], housing_labels)[1,0]))
#exit()

from sklearn.linear_model import *
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

models = [LinearRegression(),
          KNeighborsRegressor(n_neighbors=5),
          DecisionTreeRegressor(random_state=42),
          RandomForestRegressor(n_estimators=10, random_state=42),
          LassoLars(alpha=.1),
          Ridge(alpha=.5),
         ]

print("Learning: fitting the models to data...")
for m in models:
    m.fit(housing_prepared, housing_labels)
print("_______________________________________\n")

print("Evaluating the models on training set")
from sklearn.metrics import mean_squared_error
for reg in models:
    housing_predictions = reg.predict(housing_prepared)
    rmse = np.sqrt(mean_squared_error(housing_labels, housing_predictions))
    print("{:23s}: {:.3f}".format(reg.__class__.__name__, rmse))
print("_______________________________________\n")
#exit()

print("Evaluating the models using internal cross-validation")
from sklearn.model_selection import cross_val_score
for reg in models:
    scores = cross_val_score(reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
    rmse_scores = np.sqrt(-scores)
    print("{:23s}: {:.3f}+-{:.3f}".format(reg.__class__.__name__, rmse_scores.mean(), rmse_scores.std()))
print("_______________________________________\n")

print("Fine tuning the models with internal cross-validation")
from sklearn.model_selection import GridSearchCV
param_grid = [
              {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
              {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
             ]
grid_search = GridSearchCV(models[3], param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_labels)

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

print(grid_search.best_params_)

feature_importances = grid_search.best_estimator_.feature_importances_
for i in sorted(zip(feature_importances, list(housing_extra_attribs)), reverse=True):
    print(i)
print("_______________________________________\n")
#exit()

print("Final testing...")
final_model = grid_search.best_estimator_
X_test = T.drop("median_house_value", axis=1)
y_test = T["median_house_value"].copy()
X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
final_rmse = np.sqrt(mean_squared_error(y_test, final_predictions))
print("RMSE:{:10.2f}".format(final_rmse))
print("_______________________________________\n")
