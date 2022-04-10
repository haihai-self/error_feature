from sklearn import metrics
import pandas as pd

def evaluation(y, y_pre):
    top1 = metrics.accuracy_score(y, y_pre[:, 0])

    # recall = metrics.recall_score()

    return top1

