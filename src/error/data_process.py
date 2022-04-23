import pandas as pd

def processData(df):
    # sel_feature = ['classify', 'untrained_acc', 'mue_ED0', 'var_ED0', 'mue_ED', 'NMED', 'var_ED', 'mue_RED', 'var_RED', 'mue_ARED', 'var_ARED',
    #                'RMS_ED', 'RMS_RED', 'ER', 'WCE', 'WCRE']
    #
    discrete_feature = ['net', 'dataset', 'concat', 'single-sided', 'zero-error']
    #
    # result = df.loc[:, sel_feature]
    for col in discrete_feature:
        if col not in df.columns:
            continue
        temp = pd.Categorical(df[col])
        df[col] = temp.codes
    return df

def processDataSpec(df):
    sel_feature = ['classify', 'untrained_acc', 'mue_ED0', 'var_ED0', 'mue_ED', 'NMED', 'var_ED', 'mue_RED', 'var_RED', 'mue_ARED', 'var_ARED',
                   'RMS_ED', 'RMS_RED', 'ER', 'WCE', 'WCRE', 'net', 'dataset', 'concat', 'single-sided', 'zero-error']

    discrete_feature = ['single-sided', 'zero-error']

    result = df.loc[:, sel_feature]
    for col in discrete_feature:

        temp = pd.Categorical(result[col])
        result[col] = temp.codes
    return result