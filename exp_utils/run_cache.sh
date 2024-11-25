envs=(indoors)
offload_methods=(cacheinf cacheinf_local eva2_high eva2_low eva2_high_local)
# offload_methods=(local all fix flex mixed2)
# datasets=(CIFAR10 CIFAR10 OxfordIIITPet OxfordIIITPet OxfordIIITPet OxfordIIITPet)
# tasks=(classification classification segmentation segmentation detection detection)
models=(VGG19_BN ConvNeXt_Large ResNet152)
dur=300
ip=${1-"192.168.50.11"}
port=${2-"12345"}
user=${2-"user"}
wnice=${3-"enp0s31f6"}
for method in ${offload_methods[*]}; do
    for env in ${envs[*]}; do

        if [ $method == "local" ] && [ $env == "outdoors" ]
        then
            echo ""
            echo "Skipping $env $method cases..."
            continue
        fi

        echo ""
        echo "Running $env $method torchvision cases..."
        for i in {0..2}; do  # TODO run all
       if [ $method == "cacheinf" ]      # only need to add all for torchvision
        then
		continue
	fi
            if [ $method == "local" ]
            then
                bash start_work.sh $method "python3 \$work/ros_ws/src/torchvision/scripts/run_torchvision.py -a ${models[i]} -d ImageNet" $env robot2_torch13 $dur ${models[i]} False
            else
                bash start_work.sh $method "python3 \$work/ros_ws/src/torchvision/scripts/run_torchvision.py -a ${models[i]} -d ImageNet" $env robot2_torch13 $dur ${models[i]} True $user $ip $port $wnic
            fi
        done
        # fi

        echo ""
        echo "Running $env $method kapao cases..."
        if [ $method == "local" ]
        then
            bash start_work.sh $method "python3 \$work/kapao_test.py" $env robot2_torch13 $dur kapao False $user $ip $port $wnic 

            # bash start_work.sh $method "rosrun agrnav inference_ros.py" $env robot2_torch13 $dur agrnav False
        else
            bash start_work.sh $method "python3 \$work/kapao_test.py" $env robot2_torch13 $dur kapao True $user $ip $port $wnic
            
            # bash start_work.sh $method "rosrun agrnav inference_ros.py" $env robot2_torch13 $dur agrnav True  $user $ip $port
        fi 

    done
done




