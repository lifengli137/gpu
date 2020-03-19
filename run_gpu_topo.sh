#!/bin/bash
nvidia-smi topo -p2p n | grep OK | head -n 8 | awk '{$1=""; print $0}' | sed 's/ //' > /tmp/topo

g++ -o gpu_topo ./gpu/gpu_topo.cpp
order=$(./gpu_topo /tmp/topo)

if [[ $? -eq 0 ]]; then
	echo $order
	export CUDA_VISIBLE_DEVICES=$order
fi
