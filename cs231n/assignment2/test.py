# -*- coding: utf-8 -*-
 
import matplotlib.pyplot as plt
import numpy as np
import fc_net
from solver import Solver
import dataset
 
data = dataset.get_CIFAR10_data()
model = fc_net.TwoLayerNet(reg=0.9,weight_scale=1e-4)
solver = Solver(model, data,                
                lr_decay=0.95,                
                print_every=100, num_epochs=40, batch_size=400, 
                update_rule='sgd_momentum',                
                optim_config={'learning_rate': 5e-4, 'momentum': 0.5})
 
solver.train()                 
 
 
best_model = model
y_test_pred = np.argmax(best_model.loss(data['X_test']), axis=1)
y_val_pred = np.argmax(best_model.loss(data['X_val']), axis=1)
print('Validation set accuracy: ', (y_val_pred == data['y_val']).mean())
print('Test set accuracy: ', (y_test_pred == data['y_test']).mean())