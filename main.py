from sklearn.cluster import KMeans
from sklearn import neural_network

from CSVReader import CSVReader


def scaleMinMax(data,col, mini=None, maxi=None):
    minFeat = mini
    maxFeat = maxi
    for i in range(len(data)):
        data[i][col]=(float(data[i][col]) - float(minFeat)) / (float(maxFeat) - float(minFeat))


def evalMultiClass(real,computed):
    from sklearn.metrics import confusion_matrix
    conf=confusion_matrix(real,computed)
    accuracy=sum([conf[i][i] for i in range(len(set(real)))])/len(computed)
    return accuracy,conf

reader=CSVReader("iris.data")
reader.readData()
reader.splitData()


for i in range(0,4):
    mini=min([el[i] for el in reader.trainInputs])
    maxi=max([el[i] for el in reader.trainInputs])
    scaleMinMax(reader.trainInputs,i,mini,maxi)
    scaleMinMax(reader.testInputs,i,mini,maxi)



kmeans = KMeans(n_clusters=3,n_init=5)
kmeans.fit(reader.trainInputs,reader.trainOutputs)
rez=kmeans.predict(reader.testInputs)
for i in range(len(reader.testOutputs)):
    if reader.testOutputs[i]=='Iris-setosa':
        reader.testOutputs[i]=0
    elif reader.testOutputs[i]=='Iris-versicolor':
        reader.testOutputs[i]=1
    else:
        reader.testOutputs[i]=2

print(list(rez))
print(reader.testOutputs)



nn=neural_network.MLPClassifier(hidden_layer_sizes=(5,), activation='relu', max_iter=200, solver='sgd',verbose=10, random_state=1, learning_rate_init=.1)
nn.fit(reader.trainInputs,reader.trainOutputs)
rez=nn.predict(reader.testInputs)
for i in range(len(rez)):
    if rez[i]=='Iris-setosa':
        rez[i]=0
    elif rez[i]=='Iris-versicolor':
        rez[i]=1
    else:
        rez[i]=2

rez=[int(el) for el in rez]

print(evalMultiClass(reader.testOutputs,rez)[0])
