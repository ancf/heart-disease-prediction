import numpy as numpy
import pandas as pandas
import tensorflow as tensor
import keras as keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score


#set decimal precission
numpy.set_printoptions(precision=3, suppress=True)

#load dataset from *.csv
heart = pandas.read_csv("dataset/heart.csv")
print(heart)

heart["cp"] = heart["cp"].map({0:'cp type 0', 1:'cp type 1', 2:'cp type 2', 3:'cp_type 3'}) #+3 columns
heart = pandas.get_dummies(heart, columns = ["cp"], prefix='', prefix_sep='')
print(heart)

heart["restecg"] = heart["restecg"].map({0:'normal', 1:'abnormal type 1', 2:'abnormal type 2'}) #+2 columns
heart = pandas.get_dummies(heart, columns = ["restecg"], prefix='', prefix_sep='')
print(heart)

heart["slope"] = heart["slope"].map({0:'upsloping', 1:'flat', 2: 'downsloping'}) #+2 columns
heart = pandas.get_dummies(heart, columns = ["slope"], prefix='', prefix_sep='')
print(heart)
heart["thal"] = heart["thal"].map({0:'passed', 1:'failed', 2:'failed after excersise'}) #+2 columns
heart = pandas.get_dummies(heart, columns = ["thal"], prefix='', prefix_sep='')

heart = heart.reindex(columns=["age","sex","cp type 0", "cp type 1", "cp type 2", "cp_type 3","trestbps","chol","fbs","normal", "abnormal type 1", "abnormal type 2","thalach","exang","oldpeak","upsloping", "flat", "downsloping","ca","passed", "failed", "failed after excersise","target"])

print(heart)
heart.tail()
print(heart)


X = pandas.DataFrame(heart.iloc[:, 0:22].values)
y = heart.iloc[:, 22].values

print(y)


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

#print(testX)

sc = StandardScaler()
trainX = sc.fit_transform(trainX)
testX = sc.fit_transform(testX)

#print(testX)

#initializing ANN
classifier = Sequential();

#add input layer + first hidden layer 
#relu - rectifier activation function
#as many input_dims as independent variables
classifier.add(Dense(units=18, kernel_initializer = 'uniform', activation='relu', input_dim = 22))
#second hidden layer
classifier.add(Dense(18, kernel_initializer = 'uniform', activation='relu'))
#output layer
classifier.add(Dense(1, kernel_initializer = 'uniform', activation='sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(trainX, trainY, batch_size = 10, epochs = 300)

predY = classifier.predict(testX)
print(numpy.c_[predY, testY])
predY = (predY > 0.5)
print(numpy.c_[predY, testY])

cm = confusion_matrix(testY, predY)
print(cm)
accuracy_score(testY, predY)
print(accuracy_score(testY, predY))

#weights = classifier.layers[0].get_weights()[0];
#biases = classifier.layers[0].get_weights()[1];
#print(weights);
#print(biases);