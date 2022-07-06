import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocessing
from sklearn import linear_model
from sklearn.model_selection import learning_curve


# ============================= read trianing set =================================

data_train = pd.read_csv('/Users/wuruotian/Desktop/Kaggle/kaggle_project/Titanic/train.csv')
# print(data_train.info()) # dispaly columns and their data types
# print(data_train.describe()) # display statistical attributes

# ============================= read trianing set =================================


# ============================== data visualization ===============================

# ---------------- first plot ----------------
# plt.subplot2grid((2, 3), (0, 0))  # create a 2*3 convas, plot at location(0, 0)
# data_train.Survived.value_counts().plot(kind='bar')  # count number of 0 and 1, show as a bar chart
# plt.title('Survived (0 = No, 1 = Yes)')
# plt.ylabel('Number of Passangers')

# plt.subplot2grid((2, 3), (0, 1))  # plot at (0, 1)
# data_train.Pclass.value_counts().plot(kind='bar')
# plt.title('Class')
# plt.ylabel('Number of Passangers')

# # plot a scatter plot of survived against age
# plt.subplot2grid((2, 3), (0, 2))
# plt.scatter(data_train.Survived, data_train.Age)
# plt.title('Age V.S. Survived')
# plt.ylabel('Age')
# plt.grid(b=True, which='major', axis='y') # set the grid line

# plt.subplot2grid((2,3),(1,0), colspan=2)
# data_train.Age[data_train.Pclass == 1].plot(kind='kde')   # plots a kernel desnsity estimation of passanges's age from the 1st class
# data_train.Age[data_train.Pclass == 2].plot(kind='kde')
# data_train.Age[data_train.Pclass == 3].plot(kind='kde')
# plt.xlabel("Age")
# plt.ylabel("pdf") 
# plt.title("Distribution of Age of each Class")
# plt.legend(('1st class', '2nd class', '3rd class'), loc='best') # sets our legend for our graph.

# plt.subplot2grid((2,3),(1,2))
# data_train.Embarked.value_counts().plot(kind='bar')
# plt.title("Embark")
# plt.ylabel("Number of Passangers")  

# plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4) # adjust margin and space
# plt.show()
# ---------------- first plot ----------------

# ---------------- second plot ----------------
# data_Survived_1 = data_train[data_train['Survived'] == 1]  # all data entries where survived == 1
# data_Survived_0 = data_train[data_train['Survived'] == 0]
# df = pd.DataFrame({'Survived': data_Survived_1['Pclass'].value_counts(), 'Not Survived': data_Survived_0['Pclass'].value_counts()})
# df.plot(kind='bar', stacked=True)
# plt.title("Survived distribution of each Class")
# plt.xlabel("Class") 
# plt.ylabel("Number of Passangers") 

# plt.show()
# ---------------- second plot ----------------

# ---------------- third plot ----------------
# data_Survived_1 = data_train[data_train['Survived'] == 1]  # all data entries where survived == 1
# data_Survived_0 = data_train[data_train['Survived'] == 0]
# df = pd.DataFrame({'Survived': data_Survived_1['Embarked'].value_counts(), 'Not Survived': data_Survived_0['Embarked'].value_counts()})
# df.plot(kind='bar', stacked=True)
# plt.title("Survived distribution of each Embark")
# plt.xlabel("Embark") 
# plt.ylabel("Number of Passangers") 

# plt.show()
# ---------------- third plot ----------------

# ---------------- fourth plot ----------------
# data_Survived_1 = data_train[data_train['Survived'] == 1]  # all data entries where survived == 1
# data_Survived_0 = data_train[data_train['Survived'] == 0]
# df = pd.DataFrame({'Survived': data_Survived_1['Sex'].value_counts(), 'Not Survived': data_Survived_0['Sex'].value_counts()})
# df.plot(kind='bar', stacked=True)
# plt.title("Survived distribution of each Sex")
# plt.xlabel("Sex") 
# plt.ylabel("Number of Passangers") 

# plt.show()
# ---------------- fourth plot ----------------

# ---------------- fifth plot ----------------
# plt.title("Survival of Sex and Class")

# plt.subplot2grid((1, 4), (0, 0))
# data_train['Survived'][data_train['Sex'] == 'female'][data_train['Pclass'] != 3].value_counts().plot(kind='bar', color='red')
# plt.ylabel('Number of Passangers')
# plt.xlabel('high class female')

# plt.subplot2grid((1, 4), (0, 1))
# data_train['Survived'][data_train['Sex'] == 'female'][data_train['Pclass'] == 3].value_counts().plot(kind='bar', color='green')
# plt.xlabel('low class female')

# plt.subplot2grid((1, 4), (0, 2))
# data_train['Survived'][data_train['Sex'] == 'male'][data_train['Pclass'] != 3].value_counts().plot(kind='bar', color='black')
# plt.xlabel('high class male')

# plt.subplot2grid((1, 4), (0, 3))
# data_train['Survived'][data_train['Sex'] == 'male'][data_train['Pclass'] == 3].value_counts().plot(kind='bar', color='blue')
# plt.xlabel('low class male')

# plt.show()
# ---------------- fifth plot ----------------

# ---------------- sixth plot ----------------
# Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
# Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
# df=pd.DataFrame({'with cabin': Survived_cabin, 'without cabin': Survived_nocabin})
# df.plot(kind='bar', stacked=True)
# plt.title("Survived based on cabin info")
# plt.ylabel("Number of Passangers")

# plt.show()
# ---------------- sixth plot ----------------

# ============================== data visualization ===============================


# ============================== data preprocessing ===============================

# ------------- cabin --------------
# set cabin to yes and no categories
def set_cabin_value(df):
    df.loc[df.Cabin.notnull(), 'Cabin'] = 'Y'
    df.loc[df.Cabin.isnull(), 'Cabin'] = 'N'
    return df

data_train = set_cabin_value(data_train) # update Cabin value
# -------------- cabin -------------

# -------------- age --------------
# apply RandomForestClassifier to fill in missing ages
def set_missing_ages(df):
    
    # extract all numerical features, used by Random Forest Regressor
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

    # separete passanger by with and without ages
    known_age = age_df[age_df.Age.notnull()].to_numpy()
    unknown_age = age_df[age_df.Age.isnull()].to_numpy()

    # y is the target age used for training
    y = known_age[:, 0]
    
    # X is other features used for training regressor
    X = known_age[:, 1:]

    # fit RandomForestRegressor
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)
    # print('score: ', rfr.score(X, y))
    
    # use the model to predict missing ages
    predictedAges = rfr.predict(unknown_age[:, 1::])
    
    # fill in missing ages
    df.loc[df.Age.isnull(), 'Age'] = predictedAges 
    
    return df, rfr

data_train, rfr = set_missing_ages(data_train) # fill in missing ages
# print(len(data_train[pd.isnull(data_train.Age)]))
# ------------- age --------------

# ------------- categorical feature factorization --------------
# all features used for logistic regression need to be numerical values.
# e.g. for feature cabin, it takes value (yes, no), we will create two new features: cabin_yes and cabin_no
# if the entry has cabin value yes, then cabin_yes is 1 and cabin_no is 0
dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')

df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
# print(df)
# ------------- categorical feature factorization --------------

# -------------------- scaling --------------------
# In order for the logistic regression to converge faster,
# we apply scaling to those features with large range of values, so they would be limited in [-1, 1]
scaler = preprocessing.StandardScaler()

# scale age
age_reshaped = np.asarray(df['Age']).reshape(-1, 1)
age_scale_param = scaler.fit(age_reshaped)
df['Age_scaled'] = scaler.fit_transform(age_reshaped, age_scale_param)

# scale fare
fare_reshaped = np.asarray(df['Fare']).reshape(-1, 1)
fare_scale_param = scaler.fit(fare_reshaped)
df['Fare_scaled'] = scaler.fit_transform(fare_reshaped, fare_scale_param)
# print(df)
# -------------------- scaling --------------------


# ============================== data preprocessing ===============================

# ============================== Model Training ===============================

# use regular expressions to obtain all features that will be used for training
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.to_numpy()

y = train_np[:, 0]
X = train_np[:, 1:]

# fit data into logistic regression model
clf = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-6)
clf.fit(X, y)

# ============================== Model Training ===============================


# ============================== Prediction ===============================

# ----------------- preprocess test dataset -------------------
data_test = pd.read_csv('/Users/wuruotian/Desktop/Kaggle/kaggle_project/Titanic/test.csv')
data_test.loc[data_test.Fare.isnull(), 'Fare'] = 0
# print(data_test)

# fill in missing ages
temp_df = data_test[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = temp_df[temp_df.Age.isnull()].to_numpy()
X = null_age[:, 1:]
predictedAges = rfr.predict(X)
data_test.loc[data_test.Age.isnull(), 'Age'] = predictedAges

# categorical feature factorizationss
data_test = set_cabin_value(data_test)
dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')

# put all data together
df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

# scale age
age_reshaped = np.asarray(df_test['Age']).reshape(-1, 1)
age_scale_param = scaler.fit(age_reshaped)
df_test['Age_scaled'] = scaler.fit_transform(age_reshaped, age_scale_param)

# scale fare
fare_reshaped = np.asarray(df_test['Fare']).reshape(-1, 1)
fare_scale_param = scaler.fit(fare_reshaped)
df_test['Fare_scaled'] = scaler.fit_transform(fare_reshaped, fare_scale_param)

# print(df_test)
# ----------------- preprocess test dataset -------------------

# ----------------- predict and export result -----------------
test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = clf.predict(test)
result = pd.DataFrame({'PassengerId': data_test['PassengerId'].to_numpy(), 'Survived': predictions})
result.to_csv("logistic_regression_predictions.csv", index=False)
# ----------------- predict and export result -----------------

# ============================== Prediction ===============================


# ======================== Learning curve ========================

# Use sklearn.model_selection.learning_curve to get training_score and cv_score, 
# use matplotlib to plot the learning curve
"""
    estimator : Classifier we use
    title : Title
    X : input feature, type as numpy
    y : input target vector
    ylim : tuple: (ymin, ymax), highest and lowest point in y-axis
    cv : number of datasets separated in cross validation. default=3. 
        one set is used for cross validation, the others are training sets
    n_jobs : default=1, number of parallel run
"""
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, 
                        train_sizes=np.linspace(0.05, 1.0, 20), verbose=0, plot=True):

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("training size")
        plt.ylabel("score")
        plt.gca().invert_yaxis()
        plt.grid()
    
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, 
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, 
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label="trainging score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label="CV score")
    
        plt.legend(loc="best")
        
        plt.draw()
        plt.gca().invert_yaxis()
        plt.show()
    
    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff


# call above function to plot the learning curve
y = train_np[:, 0]
X = train_np[:, 1:]
# plot_learning_curve(clf, "Learning Curve", X, y)

# ======================== Learning curve ========================


# ======================== Model Optimization ========================

# display coefficient of each attribute
coefficient = pd.DataFrame({'column': list(train_df.columns)[1:], 'coefficient': list(clf.coef_.T)})
# print(coefficient)

# ======================== Model Optimization ========================