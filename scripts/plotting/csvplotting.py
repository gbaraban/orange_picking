import csv
import matplotlib.pyplot as plt
import sys
import os

filenames = [#'/home/gabe/Downloads/temp.csv']
#'/home/gabe/Downloads/run-tensorboard_2020-07-21_14-35-49-tag-train_loss.csv','/home/gabe/Downloads/run-tensorboard_2020-07-21_14-35-49-tag-val_loss.csv']
#'/home/gabe/Downloads/run-tensorboard_2020-07-21_14-35-49-tag-train_acc_0.csv','/home/gabe/Downloads/run-tensorboard_2020-07-21_14-35-49-tag-train_acc_1.csv','/home/gabe/Downloads/run-tensorboard_2020-07-21_14-35-49-tag-train_acc_2.csv']
#'/home/gabe/Downloads/run-tensorboard_2020-07-21_14-35-49-tag-val_acc_0.csv','/home/gabe/Downloads/run-tensorboard_2020-07-21_14-35-49-tag-val_acc_1.csv','/home/gabe/Downloads/run-tensorboard_2020-07-21_14-35-49-tag-val_acc_2.csv']
'/home/gabe/Downloads/run-tensorboard_2020-07-27_03-28-26-tag-train_loss.csv','/home/gabe/Downloads/run-tensorboard_2020-07-27_03-28-26-tag-val_loss.csv']
labels = ['Training','Validation']
#'Training - Point 1', 'Training - Point 2', 'Training - Point 3']
#'Validation - Point 1', 'Validation - Point 2', 'Validation - Point 3']
#xlim = (0,900)
xlab = 'Steps'
#ylab = 'Accuracy (Meters)'
ylab = 'Cross-Entropy Loss'
#ylim = (0,0.5)

x_axis = 1 #steps
y_axis = 2 #data

fig = plt.figure()
ax = plt.axes()

for fn in filenames:
  x = None
  y = None
  with open(fn,newline='') as f:
    r = csv.reader(f,delimiter=',')
    
    for d in r:
      #print(d)
      if x is None:
        x = []
        y = []
      else:
        x.append(float(d[x_axis]))
        y.append(float(d[y_axis]))
  ax.plot(x,y)

#plt.xlim(xlim)
plt.xlabel(xlab)
plt.ylabel(ylab)
#plt.ylim(ylim)
ax.legend(labels)
plt.show()
