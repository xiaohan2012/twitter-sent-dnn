#! /bin/bash

cat hyper_param_cmds.list | parallel  --progress -S ukko096,ukko095,ukko191,ukko132  --workdir . --tmpdir parallel_output --files  --jobs 4
