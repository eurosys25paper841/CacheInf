#!/bin/sh 

DOWNLINK_SPEED=$2
UPLINK_SPEED=$3
NET_DEV=$1

PKT_LIMIT=64

if [ $# -eq 1 ]; then
  sudo tc qdisc del dev $NET_DEV root
  sudo tc qdisc del dev $NET_DEV ingress
  sudo ip link del dev ifb0
  exit
fi

# uplink
sudo tc qdisc replace dev $NET_DEV root netem rate ${UPLINK_SPEED}kbit limit $PKT_LIMIT

# downlink
sudo tc qdisc add dev $NET_DEV handle ffff: ingress >/dev/null 2>&1

sudo ip link add name ifb0 type ifb >/dev/null 2>&1
sudo ip link set up ifb0
sudo tc filter replace dev $NET_DEV parent ffff: protocol ip u32 match u32 0 0 action mirred egress redirect dev ifb0

sudo tc qdisc replace dev ifb0 root netem rate ${DOWNLINK_SPEED}kbit limit $PKT_LIMIT

