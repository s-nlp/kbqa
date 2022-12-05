#!/bin/bash

python3 candidate_generation_demo.py &
python3 candidate_generation_tg_bot.py &
  
# Wait for any process to exit
wait -n
  
# Exit with status of process that exited first
exit $?