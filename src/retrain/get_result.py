import csv
import os

import pandas as pd


application = 'resnet18_mnist_origin_app'
def getFinalAcc():
    result_dir = '../../result/retrain'
    application_dir = result_dir + '/' + application
    mul_generations = os.listdir(application_dir)
    tested = set()
    if os.path.isfile('result_analyze/' + application + '.log'):
        with open('result_analyze/' + application + '.log', 'r') as f:
            for line in f.readlines():
                if line != '\n':
                    name = line.strip().split()[0]
                    tested.add(name)
    with open('result_analyze/' + application + '.log', 'a+') as f:
        for mul_generation in mul_generations:
            mul_generation_dir = application_dir + '/' + mul_generation
            mul_names = os.listdir(mul_generation_dir)
            for mul_name in mul_names:
                mul_dir = mul_generation_dir + '/' + mul_name
                if mul_name in tested:
                    continue
                if os.path.isfile(mul_dir + '/acc.log'):
                    with open(mul_dir + '/acc.log', 'r') as acc:
                        acc_retrain = acc.readline()
                        if len(acc_retrain) < 3:
                            os.remove(mul_dir + '/acc.log')
                            continue
                        acc_inf = -1
                        f.write(mul_name + ' retrain ' + str(acc_retrain) + ' without_retrain ' + str(acc_inf) + '\n')
                        print(acc_retrain)


def getResult():
    err_name = ["mul_name"]
    err_list = []
    with open('../error/source/err.txt', 'r') as f:
        err_feature = f.readlines()
        for i in range(0, len(err_feature), 2):
            mul_name = err_feature[i].strip()
            row = [mul_name]
            features = err_feature[i + 1].strip().split(";")
            for j in range(len(features) - 1):
                line = features[j]
                if line != "":
                    line_list = line.split("=")
                    if i == 0:
                        err_name.append(line_list[0])
                    row_data = line_list[1]
                    if line_list[0] == 'single-sided' or line_list[0] == 'zero-error':
                        if row_data == '0':
                            row_data = 'NO'
                        else:
                            row_data = 'YES'
                        row.append(row_data)
                    else:
                        row.append(abs(float(row_data)))
            err_list.append(row)
    err = pd.DataFrame(err_list, columns=err_name)
    err["mue_ED0"] = abs(err["mue_ED0"].astype(float))
    err.to_csv('../error/err.csv', index=False)
    res = err.set_index('mul_name')
    res.insert(0, "untrained", -1.0)
    res.insert(0, "trained", -1.0)

    with open('result_analyze/' + application + '.log', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            if line == "/n" or len(line) < 6:
                continue

            name = line[0].split(".")[0]
            trained_acc = float(line[5])
            untrained_acc = float(line[-1])
            res.loc[name, "trained"] = trained_acc
            res.loc[name, "untrained"] = untrained_acc

    res.to_csv(path_or_buf='result_analyze/res.csv')



if __name__ == '__main__':
    getFinalAcc()
    getResult()

