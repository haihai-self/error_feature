#! /bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mohai/workspace/tf-approximate/tf2/build

pidof appevaluation.py # 检测程序是否运行
while [ $? -ne 0 ]    # 判断程序上次运行是否正常结束
do
    echo "Process exits with errors! Restarting!"
    python appevaluation.py    #重启程序
done
echo "Process ends!"
