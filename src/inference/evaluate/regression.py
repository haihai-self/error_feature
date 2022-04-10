from sklearn import metrics


def evaluation(y, y_pre):
    r2 = metrics.r2_score(y, y_pre)

    acc = metrics.mean_absolute_percentage_error(y, y_pre)

    return acc, r2

def mepa_score(y, y_pre):
    return metrics.mean_absolute_percentage_error(y, y_pre)

def score():
    return metrics.make_scorer(mepa_score, greater_is_better=False)