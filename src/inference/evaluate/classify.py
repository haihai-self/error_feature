from sklearn import metrics


def evaluation(y, y_pre):
    top1 = metrics.accuracy_score(y, y_pre[:, 0])

    count = 0
    for i in range(len(y_pre)):
        if y[i] == y_pre[i][0] or y[i] == y_pre[i][1]:
            count += 1

    top2 = count / len(y_pre)

    y_recall_top1 = y.copy()
    y_pre_recall_top1 = y_pre[:, 0].copy()
    y_recall_top1[y_recall_top1 == 2] = 1
    y_pre_recall_top1[y_pre_recall_top1 == 2] = 1

    recall_1 = metrics.recall_score(y_recall_top1, y_pre_recall_top1, pos_label=0)

    weight_tpr = metrics.recall_score(y, y_pre[:, 0], average='weighted')
    macro_tpr = metrics.recall_score(y, y_pre[:, 0], average='macro')

    return top1, top2, recall_1, weight_tpr, macro_tpr

def macro_tpr(y, y_pre):
    return metrics.recall_score(y, y_pre, average='macro')


def score():

    return metrics.make_scorer(macro_tpr, greater_is_better=True)

def sel_res():
    res_c = {'chi':[],
           'var':[],
           'mrmr_c':[],
           'mrmr_dd':[],
           'mrmr_dq':[],
           'lvm_svm':[],
           'lvm_dt':[],
           'lvm_rf':[],
           'lvm_mlp':[],
            'dfr':[]
           }

    res_r = {'chi':[],
           'var':[],
           'mrmr_c':[],
           'mrmr_dd':[],
           'mrmr_dq':[],
           'lvm_svm':[],
           'lvm_dt':[],
           'lvm_rf':[],
           'lvm_mlp':[],
            'dfr':[]
           }
    return res_c, res_r