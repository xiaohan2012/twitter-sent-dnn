#! /bin/bash

program=$1
if [ -z "$program" ]; then
	echo "program name should be given as the first argument"
	exit -1
fi

username=$(whoami)

servers=$(<servers.lst)
echo "SERVERS TO STOP: "
echo $servers

IFS=',' read -ra servers <<< "$servers"

for server in "${servers[@]}"; do
    echo "Stopping tasks on $server.."
	echo "ssh $server pkill -2 -u $username $program"
    ssh $server pkill -2 -u $username $program
done


