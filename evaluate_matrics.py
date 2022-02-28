import numpy as np
from sklearn import metrics


label=[1,1,1,1,0,1,1]
label_arr=np.array(label)

out=[0,1,1,1,0,1,1]
out_arr=np.array(out)


TP=0
FN=0
FP=0
TN=0
for i in range(len(out_arr)):
    if out_arr[i]==label_arr[i]:
        if out_arr[i] == 1:
            TP+=1
        else:
            TN+=1
    else:
        if out_arr[i] == 1:
            FP+=1
        else:
            FN+=1

accuracy=(out_arr==label_arr).sum()/len(out)
precision=TP/(TP+FP)
recall=TP/(TP+FN)

print("%.3f %.3f %.3f"%(accuracy,precision,recall))

print(metrics.classification_report(label,out))
#metrics.precision_score()
