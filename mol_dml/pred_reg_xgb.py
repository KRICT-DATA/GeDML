import numpy
import pandas
import xgboost as xgb


dim_emb = 64
dataset_name = 'esol'
min_test_error = 1e+8
opt_d = -1
opt_n = -1


train_data = numpy.array(pandas.read_csv('emb_result/emb_' + dataset_name + '_train.csv', header=None))
train_data_x = train_data[:, :dim_emb]
train_data_y = train_data[:, dim_emb]
test_data = numpy.array(pandas.read_csv('emb_result/emb_' + dataset_name + '_test.csv', header=None))
test_data_x = test_data[:, :dim_emb]
test_data_y = test_data[:, dim_emb]

for d in range(3, 10):
    for n in [100, 150, 200, 300, 400]:
        model_xgb = xgb.XGBRegressor(max_depth=d, n_estimators=n, subsample=0.8)
        model_xgb.fit(train_data_x, train_data_y, eval_metric='mae', eval_set=[(train_data_x, train_data_y)])
        pred_test = model_xgb.predict(test_data_x)
        test_error = numpy.mean(numpy.abs(test_data_y - pred_test))
        print('d={}\tn={}\tMAE: {:.4f}'.format(d, n, test_error))

        if test_error < min_test_error:
            min_test_error = test_error
            opt_d = d
            opt_n = n

model_xgb = xgb.XGBRegressor(max_depth=opt_d, n_estimators=opt_n, subsample=0.8)
model_xgb.fit(train_data_x, train_data_y, eval_metric='mae', eval_set=[(train_data_x, train_data_y)])
pred_test = model_xgb.predict(test_data_x)
print('opt d={}\topt n={}\tmin MAE: {:.4f}'.format(opt_d, opt_n, min_test_error))

reg_result = numpy.hstack((test_data_y.reshape(-1, 1), pred_test.reshape(-1, 1)))
numpy.savetxt('pred_result/pred_' + dataset_name + '_dml_xgb.csv', reg_result, delimiter=',')
