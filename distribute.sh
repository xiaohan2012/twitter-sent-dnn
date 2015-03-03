#! /bin/bash


cmd_file=$1

if [ -z "$cmd_file" ]; then
	echo "cmd file should be given"
	exit -1
fi

servers=$(<servers.lst)
echo "SERVERS TO DEPLOY: "
echo $servers

cat $cmd_file | parallel  --progress -S  $servers  --workdir . --tmpdir parallel_output --files  --jobs 6


