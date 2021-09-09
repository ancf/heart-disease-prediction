import numpy as numpy
import pandas as pandas
import tensorflow as tensor
import keras as keras
import sys, getopt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

def main(argv):
    ifile = ''
    ofile = ''
    generations=0
    units=0
    try:
        opts, args = getopt.getopt(sys.argv[1:], "i:o:u:g:h",["ifile=","ofile=","units=","generations=","help"])
    except getopt.GetoptError:
        print ('Missing arguments, run program this way:') 
        print ('predict.py -i dataset.csv -o results.txt')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('Run program this way:') 
            print ('predict.py -i dataset.csv -o results.txt')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            ifile = arg
        elif opt in ("-o", "--ofile"):
            ofile = arg
        elif opt in ("-u", "--units"):
            units = arg
        elif opt in ("-g", "--generations"):
            generations = arg
    print(ifile)
    print(ofile)
    print(generations)
    print(units)
    
    tmp_output = sys.stdout #store original output



    #set decimal precission
    numpy.set_printoptions(precision=3, suppress=True)

    #set format for pandas
    pandas.set_option('display.max_rows', None)
    pandas.set_option('display.max_columns', None)
    pandas.set_option('display.width', None)
    pandas.set_option('display.max_colwidth', None)

    #load dataset from *.csv
    heart = pandas.read_csv(ifile)
  

    heart["cp"] = heart["cp"].map({0:'cp type 0', 1:'cp type 1', 2:'cp type 2', 3:'cp_type 3'}) #+3 columns
    heart = pandas.get_dummies(heart, columns = ["cp"], prefix='', prefix_sep='')
    

    heart["restecg"] = heart["restecg"].map({0:'normal', 1:'abnormal type 1', 2:'abnormal type 2'}) #+2 columns
    heart = pandas.get_dummies(heart, columns = ["restecg"], prefix='', prefix_sep='')
    

    heart["slope"] = heart["slope"].map({0:'upsloping', 1:'flat', 2: 'downsloping'}) #+2 columns
    heart = pandas.get_dummies(heart, columns = ["slope"], prefix='', prefix_sep='')
    
    heart["thal"] = heart["thal"].map({0:'passed', 1:'failed', 2:'failed after excersise'}) #+2 columns
    heart = pandas.get_dummies(heart, columns = ["thal"], prefix='', prefix_sep='')

    heart = heart.reindex(columns=["age","sex","cp type 0", "cp type 1", "cp type 2", "cp_type 3","trestbps","chol","fbs","normal", "abnormal type 1", "abnormal type 2","thalach","exang","oldpeak","upsloping", "flat", "downsloping","ca","passed", "failed", "failed after excersise","target"])

   
    heart.tail()
    print("Input with arguments converted to numerical:", file=open(ofile, 'w'))
    print(heart, file=open(ofile, 'a'))


    X = pandas.DataFrame(heart.iloc[:, 0:22].values)
    y = heart.iloc[:, 22].values
    

    trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state = 0)

    sc = StandardScaler()
    trainX = sc.fit_transform(trainX)
    testX = sc.fit_transform(testX)

    

    #initializing ANN
    classifier = Sequential();

    #add input layer + first hidden layer 
    #relu - rectifier activation function
    #as many input_dims as independent variables
    classifier.add(Dense(units=units, kernel_initializer = 'uniform', activation='relu', input_dim = 22))
    #second hidden layer
    classifier.add(Dense(units, kernel_initializer = 'uniform', activation='relu'))
    #output layer
    classifier.add(Dense(1, kernel_initializer = 'uniform', activation='sigmoid'))

    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    classifier.fit(trainX, trainY, batch_size = 10, epochs=int(generations))

    print("Original confucion matrix:", file=open(ofile, 'a'))
    predY = classifier.predict(testX)
    print(numpy.c_[predY, testY], file=open(ofile, 'a'))
    print("Confusion matrix (rounded):", file=open(ofile, 'a'))
    predY = (predY > 0.5)
    print(numpy.c_[predY, testY], file=open(ofile, 'a'))

    cm = confusion_matrix(testY, predY)
    print("Accuracy score:", file=open(ofile, 'a'))
    print(cm, file=open(ofile, 'a'))
    accuracy_score(testY, predY)
      
    sys.stdout = tmp_output
    
    print(accuracy_score(testY, predY))

    #weights = classifier.layers[0].get_weights()[0];
    #biases = classifier.layers[0].get_weights()[1];
    #print(weights);
    #print(biases);


if __name__ == "__main__":
    main(sys.argv[1:])