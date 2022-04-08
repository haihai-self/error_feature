import pandas as pd

if __name__ == '__main__':
    res = pd.read_csv('resnet18_cifar10.csv')
    res = res.rename(columns={'trained':'resnet18_cifar10'})
    f1 = pd.read_csv('resnet18_mnist.csv')[['mul_name', 'trained']]
    res = res.merge(f1, on=['mul_name'])
    res.insert(1, 'resnet18_mnist', res['trained'])
    res = res.drop(columns=['trained'])
    res.to_csv('final_table.csv', index=False)