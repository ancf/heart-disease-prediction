import numpy as numpy
import pandas as pandas
import tensorflow as tensor
import keras as keras
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

#set decimal precission
numpy.set_printoptions(precision=3, suppress=True)

#load dataset from *.csv
heart = pandas.read_csv("dataset/heart.csv")
heart["sex"] = heart["sex"].map({0:'female', 1:'male'})
heart = pandas.get_dummies(heart, columns = ["sex"], prefix='', prefix_sep='')

heart["cp"] = heart["cp"].map({0:'cp type 0', 1:'cp type 1', 2:'cp type 2', 3:'cp_type 3'})
heart = pandas.get_dummies(heart, columns = ["cp"], prefix='', prefix_sep='')

heart["fbs"] = heart["fbs"].map({0:'fasting blood sugar normal', 1:'fasting blood sugar too high'})
heart = pandas.get_dummies(heart, columns = ["fbs"], prefix='', prefix_sep='')

heart["restecg"] = heart["restecg"].map({0:'normal', 1:'abnormal type 1', 2:'abnormal type 2'})
heart = pandas.get_dummies(heart, columns = ["restecg"], prefix='', prefix_sep='')

heart["exang"] = heart["exang"].map({0:'absent', 1:'present'})
heart = pandas.get_dummies(heart, columns = ["exang"], prefix='', prefix_sep='')

heart["slope"] = heart["slope"].map({0:'upsloping', 1:'flat', 2: 'downsloping'})
heart = pandas.get_dummies(heart, columns = ["slope"], prefix='', prefix_sep='')

heart["thal"] = heart["thal"].map({0:'passed', 1:'failed', 2:'failed after excersise'})
heart = pandas.get_dummies(heart, columns = ["thal"], prefix='', prefix_sep='')

heart["target"] = heart["target"].map({0:'healthy', 1:'unhealthy'})
heart = pandas.get_dummies(heart, columns = ["target"], prefix='', prefix_sep='')
heart.tail()
print(heart)


X = pandas.DataFrame(heart.iloc[:, 0:25].values)
y = heart.iloc[:, 25:26].values

print(X)

#onehotencoder = OneHotEncoder()

#labelencoder_X_2 = LabelEncoder()
#X.loc[:, 1] = labelencoder_X_2.fit_transform(X.iloc[:, 1]) #sex
#X.loc[:, 2] = labelencoder_X_2.fit_transform(X.iloc[:, 2]) #cp
#X.loc[:, 5] = labelencoder_X_2.fit_transform(X.iloc[:, 5]) #fbs
#X.loc[:, 6] = labelencoder_X_2.fit_transform(X.iloc[:, 6]) #restecg
#X.loc[:, 8] = labelencoder_X_2.fit_transform(X.iloc[:, 8]) #exang
#X.loc[:, 10] = labelencoder_X_2.fit_transform(X.iloc[:, 10]) #slope
#X.loc[:, 12] = labelencoder_X_2.fit_transform(X.iloc[:, 12]) #thal
#X = onehotencoder.fit_transform(X).toarray()
#y = labelencoder_X_2.fit_transform(y) #target

#X = onehotencoder.fit_transform(X).toarray()
#X = X[:, 1:]
#X = X[:, 2:]
#X = X[:, 5:]
#X = X[:, 6:]
#X = X[:, 8:]
#X = X[:, 10:]
#X = X[:, 12:]

#X = X[:, 1:] 


#train and test sets
#train_dataset = heart.sample(frac=0.8, random_state=0)
#test_dataset = heart.drop(train_dataset.index)


trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state = 0)

print(testX)

sc = StandardScaler()
trainX = sc.fit_transform(trainX)
testX = sc.fit_transform(testX)

print(testX)
