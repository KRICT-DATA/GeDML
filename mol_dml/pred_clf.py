import numpy
import pandas
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef


dataset_name = 'bace'
dim_emb = 64

train_data = numpy.array(pandas.read_csv('emb_result/emb_' + dataset_name + '_train.csv', header=None))
train_data_x = train_data[:, :dim_emb]
train_data_y = train_data[:, dim_emb]
test_data = numpy.array(pandas.read_csv('emb_result/emb_' + dataset_name + '_test.csv', header=None))
test_data_x = test_data[:, :dim_emb]
test_data_y = test_data[:, dim_emb]

for k in range(3, 21):
    print('----------------' + str(k) + '----------------------------')
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_data_x, train_data_y)
    pred_test = knn.predict(test_data_x)
    print(numpy.sum(test_data_y == pred_test) / float(test_data_y.shape[0]))
    print(f1_score(test_data_y, pred_test, average='weighted', labels=numpy.unique(pred_test)))
    print(matthews_corrcoef(test_data_y, pred_test))
