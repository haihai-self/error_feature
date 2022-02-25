#! /bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mo/workspace/mh/pycharm/tf-approximate/tf2/build


pidof vgg16app3090.py # 检测程序是否运行

function runApp() {
    echo "run $1"
#    echo $temp
    pidof ../retrain/vgg16app3090.py # 检测程序是否运行

    while [ $? -ne 0 ]
    do
      python ../retrain/vgg16app3090.py $1 &> shellLog/$1.log
#      python ../retrain/vgg16app3090.py $1

    done

}
for comd in "mnist" "cifar10"
do
  echo $comd
  {
    runApp $comd
  }&
done
wait
echo "Process ends!"
