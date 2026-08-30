#!/bin/bash

#SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

export CARLA_ROOT="$(realpath "${SCRIPT_DIR}/../../../carla")"
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh

if [ ! -x "${CARLA_SERVER}" ]; then
    echo "Cannot find Carla Server：${CARLA_SERVER}"
    exit 1
fi

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:scenario_runner

export LEADERBOARD_ROOT=leaderboard
export CHALLENGE_TRACK_CODENAME=SENSORS
# export PORT=2000
export TM_PORT=2500
export REPETITIONS=1 # multiple evaluation runs, Never changed
export RESUME=True 
export DEBUG_CHALLENGE=0  # TODO: Check
# export DEBUG_CHALLENGE=1
export DATA_COLLECTION=True # TODO: Check
export MAX_SPEED=5
# export MAX_SPEED=15



# Roach data collection


MODEL=$1
if [ "$MODEL" = "TCP" ]; then
    echo "TCP"
    export TEAM_AGENT=leaderboard/leaderboard/tcp_agent.py
    export TEAM_CONFIG=TCP/epoch=59-last.ckpt
elif [ "$MODEL" = "InterFuser" ]; then
    echo "InterFuser"
    export TEAM_AGENT=leaderboard/team_code/interfuser_agent.py # agent
    export TEAM_CONFIG=leaderboard/team_code/interfuser_config.py # model checkpoint, not required for expert
else
    echo "Please include the ADS to be tested (TCP, InterFuser)"
    exit 1
fi


SECTION=$2
if [ "$SECTION" = "Curve" ]; then
    echo "routes_courve"
    export ROUTE_FILE=routes_courve
elif [ "$SECTION" = "Straight" ]; then
    echo "routes_short"
    export ROUTE_FILE=routes_short
else
    echo "Please include the road section (Curve, Straight)"
    exit 1
fi



## Experiment Controls
export AGENT_MODE=0
export ADS_MODEL=False
export GA=False
export SURROGATE=False
export SURROGATE_MODEL=None
export TIMEOUT=120
# export TIMEOUT=10

LEVEL=$3
if [ "$LEVEL" = "0" ]; then
    echo "Density Level 0"
    export REGION=0
elif [ "$LEVEL" = "1" ]; then
    echo "Density Level 1"
    export REGION=28
elif [ "$LEVEL" = "2" ]; then
    echo "Density Level 2"
    export REGION=21
elif [ "$LEVEL" = "3" ]; then
    echo "Density Level 3"
    export REGION=14
elif [ "$LEVEL" = "4" ]; then
    echo "Density Level 4"
    export REGION=7
elif [ "$LEVEL" = "baseline" ]; then
    echo "Density Level baseline"
    export REGION=-1
else
    echo "Please include the road section (Curve, Straight)"
    exit 1
fi


VECTOR_PATH=$4

CURRENT_TIME=$5
# current_time=$(date "+%Y-%m-%d|%H:%M:%S")
export ROUTES=leaderboard/data/TCP_training_routes/${ROUTE_FILE}.xml
export CHECKPOINT_ENDPOINT=data_collect_${ROUTE_FILE}_${CURRENT_TIME}.json
export SCENARIOS=leaderboard/data/scenarios/all_towns_traffic_scenarios.json

# PORT=$6
export PORT=$6
# echo "PORT1: $PORT"

## Information Collection
# export SAVE_IMG=True
export SAVE_IMG=False
export LOG=False
# SAVE_IMG_TEXT=$(if [ "$SAVE_IMG" = "True" ]; then echo "SAVE_IMG"; else echo "NONE_IMG"; fi)
# export SAVE_PATH=../SBT-data/${MODEL}/${CURRENT_TIME}--${SAVE_IMG_TEXT}/

export SAVE_ADS=./data/${MODEL}
mkdir -p "${SAVE_ADS}"
export SAVE_PATH=./data/${MODEL}/${CURRENT_TIME}/


# python3 ${LEADERBOARD_ROOT}/leaderboard/run_one_case.py \
python3 ${LEADERBOARD_ROOT}/leaderboard/search.py \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--port=${PORT} \
--fitness_path=${SAVE_PATH}/fitness.csv \
--agent_mode=${AGENT_MODE} \
--trafficManagerPort=${TM_PORT} \
--timeout=${TIMEOUT} \
--vector_path=${VECTOR_PATH}
