mode=${1-"flex"}    # flex means no latency constraint; fix means 1 hz latency constraint
cmd=${2-"rosrun kapao pose_follower.py"}
env=${3-"indoors"}
container=${4-"robot2"}
dur=${5-1800}
workload=${6-"kapao"}
offload=${7-"True"}
username=${8-"user"}
ip=${9-"192.168.50.11"}
port=${10-"12345"}
wnic=${11-"enp0s31f6"}
log_dir="${mode}_${env}_${workload}"
bag_name=${12-"all_topic.bag"}
echo "offload_mode: $mode; env: $env; cmd: $cmd; container: $container; offload: $offload"
echo "Do start replay bandwidth at $env.txt at the server side."

echo "log_dir $work/log/$log_dir"

docker restart $container &> /dev/null
echo "Restarted container for robot control and native torch"

tmux has-session -t offload_exp 2>/dev/null

if [ $? != 0 ]; then
  # Set up your session
    tmux new -t offload_exp -d

    sleep 5
    tmux new-window -t offload_exp -n env
    tmux new-window -t offload_exp -n work
    tmux new-window -t offload_exp -n power
    tmux new-window -t offload_exp -n roscore
    tmux new-window -t offload_exp -n rosbag
    sleep 3
    tmux send-keys -t offload_exp:power "sudo su" ENTER
fi

tmux send-keys -t offload_exp:roscore C-c ENTER; tmux send-keys -t offload_exp:roscore "rosclean purge -y" ENTER; 
tmux send-keys -t offload_exp:rosbag C-c ENTER
tmux send-keys -t offload_exp:roscore "roscore" ENTER
sleep 5
tmux send-keys -t offload_exp:rosbag "rosparam set /use_sim_time true; cd; rosbag play -l --clock $work/$bag_name" ENTER

ssh $username@$ip "tmux has-session -t replay_bw; if [ \$? != 0 ]; then tmux new -t replay_bw -d; sleep 3; tmux new-window -t replay_bw -n offload_server; tmux new-window -t replay_bw -n server_host; tmux new-window -t replay_bw -n bw; sleep 2; fi"
sleep 10
ssh $username@$ip "tmux send-keys -t replay_bw:server C-c ENTER; tmux send-keys -t replay_bw:server_host C-c ENTER; tmux send-keys -t replay_bw:offload_server C-c ENTER; tmux send-keys -t replay_bw:server_host \"cd \\\$work; mkdir -p \\\$work/log/$log_dir; python3 start_server.py 0.0.0.0 12345 &> \\\$work/log/$log_dir/server.log\" ENTER;"

echo "Started server"
# tmux send-keys -t offload_exp:env "docker exec -it $container zsh" ENTER
# tmux send-keys -t offload_exp:env "roslaunch turn_on_wheeltec_robot ros_torch_env.launch" ENTER
# echo "Started wheeltech robot sensors and motion controller"

tmux send-keys -t offload_exp:work "docker exec -it $container zsh" ENTER
tmux send-keys -t offload_exp:work "rm -rf \$work/log/$log_dir; mkdir -p \$work/log/$log_dir; rosparam set /offload_mode $mode; rosparam set /server_ip $ip; rosparam set /server_port $port; rosparam set /offload_method $mode; rosparam set /offload_port $port; rosparam set /offload $offload" ENTER
tmux send-keys -t offload_exp:work "ROS_LOG_DIR=\$work/log/$log_dir $cmd" ENTER
echo "Running $cmd"

echo "Waiting for workload initialization."
sleep 2
while ! rosservice list | grep /Start &> /dev/null
do
    sleep 5
done
sleep 2

while ! rosservice list | grep /Start &> /dev/null
do
    sleep 5
done
sleep 2

while ! rosservice list | grep /Start &> /dev/null
do
    sleep 5
done

echo "Worload initialization finished. Press any key to start."
# read start

tmux send-keys -t offload_exp:power C-c ENTER
tmux send-keys -t offload_exp:power "python3 $work/exp_utils/power_monitor.py _interval:=1 &> $work/log/$log_dir/power_record.txt" ENTER
echo "Started power_monitor"


ssh $username@$ip "tmux send-keys -t replay_bw:bw C-c ENTER; tmux send-keys -t replay_bw:bw \"cd \\\$work/exp_utils; python3 replay_bandwidth.py ${env}.txt $wnic 1\" ENTER;"

# tmux new-window -t offload_exp -n bw
# tmux send-keys -t offload_exp:bw "python3 $work/exp_utils/replay_bandwidth.py $work/exp_utils/$env.txt &> $work/log/$log_dir/replay_bandwidth.txt" ENTER
# echo "Started replay_bandwidth.py"

rosservice call /Start
sleep 1
rosservice call /Start
sleep 1
rosservice call /Start
sleep 1
echo "Worload started. Killing in ${dur}s."

sleep 4
tmux send-keys -t offload_exp:rosbag C-c ENTER
sleep $dur
#rosnode kill -a
# tmux send-keys -t offload_exp:env C-c ENTER

tmux send-keys -t offload_exp:work C-c ENTER
tmux send-keys -t offload_exp:work C-c ENTER
tmux send-keys -t offload_exp:work C-c ENTER
tmux send-keys -t offload_exp:power C-c ENTER
tmux send-keys -t offload_exp:power C-c ENTER
tmux send-keys -t offload_exp:power C-c ENTER
docker stop $container &> /dev/null
ssh $username@$ip "tmux send-keys -t replay_bw:server C-c ENTER; tmux send-keys -t replay_bw:server_host C-c ENTER; tmux send-keys -t replay_bw:offload_server C-c ENTER; tmux send-keys -t replay_bw:bw C-c ENTER;"
ssh $username@$ip "tmux send-keys -t replay_bw:server C-c ENTER; tmux send-keys -t replay_bw:server_host C-c ENTER; tmux send-keys -t replay_bw:offload_server C-c ENTER; tmux send-keys -t replay_bw:bw C-c ENTER;"
ssh $username@$ip "tmux send-keys -t replay_bw:server C-c ENTER; tmux send-keys -t replay_bw:server_host C-c ENTER; tmux send-keys -t replay_bw:offload_server C-c ENTER; tmux send-keys -t replay_bw:bw C-c ENTER;"
sudo pkill -f "power_monitor.py"
sleep 1
sudo pkill -9 -f "power_monitor.py"
sleep 1
sudo pkill -9 -f "power_monitor.py"
# tmux send-keys -t offload_exp:bw C-c ENTER
sleep 5

echo "Fin"
echo ""

