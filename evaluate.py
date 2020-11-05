import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score as auc_score, roc_curve
import os

results_dir = '/scratch_net/bmicdl04/kender/Projects/ProstateQC/models'

# Evaluation on training set first
acc = []
for j in range(1,11):
    pr = np.loadtxt(os.path.join(results_dir, 'model_run_{}predictions.txt'.format(j)))
    gt = np.loadtxt(os.path.join(results_dir, 'model_run_{}labels.txt'.format(j)))
    acc_ = accuracy_score(gt, pr>0.0)
    acc.append(acc_)
print(np.mean(acc), np.median(acc), np.std(acc))

auc = []
for j in range(1,11):
    pr = np.loadtxt(os.path.join(results_dir, 'model_run_{}predictions.txt'.format(j)))
    gt = np.loadtxt(os.path.join(results_dir, 'model_run_{}labels.txt'.format(j)))
    auc_ = auc_score(gt, pr)
    auc.append(auc_)
print(np.mean(auc), np.median(auc), np.std(auc))

## VALIDATION DATA from the same center: Average of 10 models is the final prediction
print('====USZ Hold-out data====')
preds = []
for j in range(1,11):
    pr = np.loadtxt(os.path.join(results_dir, 'model_run_{}_predictions_holdOutSet.txt'.format(j)))
    preds.append(pr)
preds = np.asarray(preds)
mean_preds = preds.mean(axis=0)
prob_mean_preds = 1 / (1 + np.exp(-mean_preds))
gt = np.loadtxt(os.path.join(results_dir, 'model_run_1_labels_holdOutSet.txt'))
acc = accuracy_score(gt,prob_mean_preds>0.5)
print('acc with mean predictions: ', acc)
auc = auc_score(gt, prob_mean_preds)
print('auc with mean predictions: ', auc)
fpr, tpr, _ = roc_curve(gt, prob_mean_preds)
plt.figure(figsize=[8,5])
plt.plot(fpr, tpr, linewidth=3)
plt.grid(True)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig(os.path.join(results_dir, 'hold_out_roc_at_mean.png'))
plt.close()
np.savetxt(os.path.join(results_dir, 'model_runs_average_predictions_holdOutSet.txt'),
           prob_mean_preds)


## VALIDATION DATA from differen center: Average of 10 models is the final prediction
print('====WOLLISHOFEN DATA====')
preds = []
for j in range(1,11):
    pr = np.loadtxt(os.path.join(results_dir, 'model_run_{}_predictions_WollishofenData.txt'.format(j)))
    preds.append(pr)
preds = np.asarray(preds)
mean_preds = preds.mean(axis=0)
prob_mean_preds = 1 / (1 + np.exp(-mean_preds))
gt = np.loadtxt(os.path.join(results_dir, 'model_run_1_labels_WollishofenData.txt'))
acc = accuracy_score(gt,prob_mean_preds>0.5)
print('acc with mean predictions: ', acc)
auc = auc_score(gt, prob_mean_preds)
print('auc with mean predictions: ', auc)
fpr, tpr, _ = roc_curve(gt, prob_mean_preds)
plt.figure(figsize=[8,5])
plt.plot(fpr, tpr, linewidth=3)
plt.grid(True)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig(os.path.join(results_dir, 'wollishofen_roc_at_mean.png'))
plt.close()
np.savetxt(os.path.join(results_dir, 'model_runs_average_predictions_WollishofenData.txt'),
           prob_mean_preds)

