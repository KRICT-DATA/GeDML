import numpy
import random
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import util.mol_conv as mconv
import util.mol_dml as mdml
import model.GCN_DML as dml
import model.GCN as gcn
import util.trainer as tr
from torch_geometric.data import DataLoader
from sklearn.manifold import TSNE


def normalize(X):
    if len(X.shape) == 1:
        min = numpy.min(X)
        max = numpy.max(X)

        for i in range(0, X.shape[0]):
            X[i] = (X[i] - min) / (max - min)
    else:
        mins = numpy.min(X, axis=0)
        maxs = numpy.max(X, axis=0)

        for i in range(0, X.shape[0]):
            for j in range(0, X.shape[1]):
                X[i, j] = (X[i, j] - mins[j]) / (maxs[j] - mins[j])

    return X


dataset_name = 'bbbp'
batch_size = 64
max_epochs = 1000
dim_emb = 64
alpha = 0.8


data_list, smiles = mconv.read_dataset('data/classification/' + dataset_name + '.csv')
random.shuffle(data_list)

num_train_data = int(len(data_list) * 0.8)
train_data_list = data_list[:num_train_data]
test_data_list = data_list[num_train_data:]
train_data_loader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(test_data_list, batch_size=batch_size)


model = dml.GCN(mconv.num_atm_feats, dim_emb).cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

for epoch in range(0, max_epochs):
    train_loss = mdml.train_clf(model, optimizer, train_data_loader, alpha)
    print('Epoch [{}/{}]\tTrain loss: {:.4f}'.format(epoch + 1, max_epochs, train_loss))

train_data_loader = DataLoader(train_data_list, batch_size=batch_size)
emb_train = mdml.test(model, train_data_loader).cpu().numpy()
train_data_y = numpy.array([x.y.item() for x in train_data_list]).reshape((-1, 1))
emb_test = mdml.test(model, test_data_loader).cpu().numpy()
test_data_y = numpy.array([x.y.item() for x in test_data_list]).reshape((-1, 1))

numpy.savetxt('emb_result/emb_' + dataset_name + '_train.csv', numpy.hstack((emb_train, train_data_y)), delimiter=',')
numpy.savetxt('emb_result/emb_' + dataset_name + '_test.csv', numpy.hstack((emb_test, test_data_y)), delimiter=',')
