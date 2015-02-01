#! /bin/bash

cat hyper_param_cmds.list | parallel  --progress -S ukko191,ukko132,ukko136,ukko017,ukko114,ukko189,ukko139   --workdir . --tmpdir parallel_output --files  --jobs 5
