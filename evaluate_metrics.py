import numpy as np
# from sklearn.metrics import accuracy_score

y_pred=[0,2,1,3]
y_true=[0,1,2,3]
y_pred_np=np.array(y_pred)
y_true_np=np.array(y_true)
accuracy=(y_pred_np==y_true_np).sum()/len(y_pred_np)


