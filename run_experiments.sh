

#!/bin/bash 

echo "Starting script" 
if [ "$#" -lt 2 ]; then # exit if called with no arguments 
    # echo "Usage: bash $0 <DATASET NAME> <CONDA ENVIRONMENT NAME>"
    echo "Usage: bash $0 <DATASET NAME> <CONDA ENVIRONMENT NAME>"
    exit 1
fi 

DATASET=$1
CONDA_ENV_DIR="/Users/gadmohamed/miniforge3/envs"
CONDA_ENV="$CONDA_ENV_DIR/$2"

CODE_PATH="src/main.py"


source /opt/anaconda3/etc/profile.d/conda.sh
conda init zsh
conda activate $CONDA_ENV 

# REQUIREMENTS_FILE="src/requirements.txt"
# if [ -f "$REQUIREMENTS_FILE" ]; then
#     echo "Found $REQUIREMENTS_FILE file. Installing dependencies..."
#     pip install -r "$REQUIREMENTS_FILE"
#     echo "Done installing dependencies." 
# else
#     echo "No $REQUIREMENTS_FILE file found."
# fi


echo "Begin experiments!" 
echo "Code path: $CODE_PATH"
echo "Dataset: $DATASET"
echo "Conda environment: $CONDA_ENV"





# python $CODE_PATH $DATASET --learning_algorithm 'fedprox' --rounds 10  --local_epochs 1  --target_model='nn' --dp_epsilon 100 --dp_type 'dp'
# python $CODE_PATH $DATASET --learning_algorithm 'fedsgd' --rounds 10 --use_dp --local_epochs 1  --target_model='nn' --dp_epsilon 100 --dp_type 'dp'

# python $CODE_PATH $DATASET --learning_algorithm 'fedprox' --rounds 10 --local_epochs 1  --target_model='nn' 
# python $CODE_PATH $DATASET --learning_algorithm 'fedsgd' --rounds 10 --local_epochs 1  --target_model='nn'

# python $CODE_PATH $DATASET --learning_algorithm 'local' --local_epochs 40  --target_model='nn' 
# python $CODE_PATH $DATASET --learning_algorithm 'local' --local_epochs 40  --use_dp --target_model='nn'  --dp_epsilon 100 --dp_type 'dp'
# python $CODE_PATH $DATASET --learning_algorithm 'local' --local_epochs 40  --use_dp --lr 0.1 --target_model='nn' --dp_epsilon 100 --dp_type 'dp'

python $CODE_PATH $DATASET --learning_algorithm 'fedavg' --use_dp --rounds 20 --local_epochs 1 --target_model='nn' --dp_epsilon 1 --dp_type 'dp' --lr 0.1
python $CODE_PATH $DATASET --learning_algorithm 'fedavg' --use_dp --rounds 20 --local_epochs 1 --target_model='nn' --dp_epsilon 10 --dp_type 'dp' --lr 0.1
python $CODE_PATH $DATASET --learning_algorithm 'fedavg' --use_dp --rounds 20 --local_epochs 1 --target_model='nn' --dp_epsilon 100 --dp_type 'dp' --lr 0.1
python $CODE_PATH $DATASET --learning_algorithm 'fedavg' --use_dp --rounds 20 --local_epochs 1 --target_model='nn' --dp_epsilon 1000 --dp_type 'dp' --lr 0.1
python $CODE_PATH $DATASET --learning_algorithm 'fedavg' --use_dp --rounds 20 --local_epochs 1 --target_model='nn' --dp_epsilon 1 --dp_type 'rdp' --lr 0.1
python $CODE_PATH $DATASET --learning_algorithm 'fedavg' --use_dp --rounds 20 --local_epochs 1 --target_model='nn' --dp_epsilon 10 --dp_type 'rdp' --lr 0.1
python $CODE_PATH $DATASET --learning_algorithm 'fedavg' --use_dp --rounds 20 --local_epochs 1 --target_model='nn' --dp_epsilon 100 --dp_type 'rdp' --lr 0.1
python $CODE_PATH $DATASET --learning_algorithm 'fedavg' --use_dp --rounds 20 --local_epochs 1 --target_model='nn' --dp_epsilon 1000 --dp_type 'rdp' --lr 0.1

python $CODE_PATH $DATASET --learning_algorithm 'fedavg' --use_dp --rounds 20 --local_epochs 1 --target_model='nn' --dp_epsilon 1 --dp_type 'dp' --lr 0.01
python $CODE_PATH $DATASET --learning_algorithm 'fedavg' --use_dp --rounds 20 --local_epochs 1 --target_model='nn' --dp_epsilon 10 --dp_type 'dp' --lr 0.01
python $CODE_PATH $DATASET --learning_algorithm 'fedavg' --use_dp --rounds 20 --local_epochs 1 --target_model='nn' --dp_epsilon 100 --dp_type 'dp' --lr 0.01
python $CODE_PATH $DATASET --learning_algorithm 'fedavg' --use_dp --rounds 20 --local_epochs 1 --target_model='nn' --dp_epsilon 1000 --dp_type 'dp' --lr 0.01
python $CODE_PATH $DATASET --learning_algorithm 'fedavg' --use_dp --rounds 20 --local_epochs 1 --target_model='nn' --dp_epsilon 1 --dp_type 'rdp' --lr 0.01
python $CODE_PATH $DATASET --learning_algorithm 'fedavg' --use_dp --rounds 20 --local_epochs 1 --target_model='nn' --dp_epsilon 10 --dp_type 'rdp' --lr 0.01
python $CODE_PATH $DATASET --learning_algorithm 'fedavg' --use_dp --rounds 20 --local_epochs 1 --target_model='nn' --dp_epsilon 100 --dp_type 'rdp' --lr 0.01
python $CODE_PATH $DATASET --learning_algorithm 'fedavg' --use_dp --rounds 20 --local_epochs 1 --target_model='nn' --dp_epsilon 1000 --dp_type 'rdp' --lr 0.01
python $CODE_PATH $DATASET --learning_algorithm 'fedavg'  --rounds 20  --local_epochs 1  --target_model='nn' --lr 0.01
python $CODE_PATH $DATASET --learning_algorithm 'fedavg'  --rounds 20  --local_epochs 1  --target_model='nn' --lr 0.001
python $CODE_PATH $DATASET --learning_algorithm 'fedavg'  --rounds 20  --local_epochs 1  --target_model='nn' --lr 0.0001



python $CODE_PATH $DATASET --id 10 --learning_algorithm 'central' --local_epochs 20  --target_model='nn' --dp_epsilon 1 --dp_type 'dp' --lr 0.01
python $CODE_PATH $DATASET --id 11 --learning_algorithm 'central' --use_dp --local_epochs 20  --target_model='nn' --dp_epsilon 10 --dp_type 'dp' --lr 0.01
python $CODE_PATH $DATASET --id 12 --learning_algorithm 'central' --use_dp --local_epochs 20  --target_model='nn' --dp_epsilon 100 --dp_type 'dp' --lr 0.01
python $CODE_PATH $DATASET --id 12 --learning_algorithm 'central' --use_dp --local_epochs 20  --target_model='nn' --dp_epsilon 1000 --dp_type 'dp' --lr 0.01

python $CODE_PATH $DATASET --id 10 --learning_algorithm 'central' --local_epochs 20  --target_model='nn' --dp_epsilon 1 --dp_type 'dp' --lr 0.001
python $CODE_PATH $DATASET --id 11 --learning_algorithm 'central' --use_dp --local_epochs 20  --target_model='nn' --dp_epsilon 10 --dp_type 'dp' --lr 0.001
python $CODE_PATH $DATASET --id 12 --learning_algorithm 'central' --use_dp --local_epochs 20  --target_model='nn' --dp_epsilon 100 --dp_type 'dp' --lr 0.001
python $CODE_PATH $DATASET --id 12 --learning_algorithm 'central' --use_dp --local_epochs 20  --target_model='nn' --dp_epsilon 1000 --dp_type 'dp' --lr 0.001

python $CODE_PATH $DATASET --id 10 --learning_algorithm 'central' --local_epochs 20  --target_model='nn' --lr 0.01
python $CODE_PATH $DATASET --id 11 --learning_algorithm 'central' --use_dp --local_epochs 20  --target_model='nn' --dp_epsilon 10 --dp_type 'rdp' --lr 0.01
python $CODE_PATH $DATASET --id 12 --learning_algorithm 'central' --use_dp --local_epochs 20  --target_model='nn' --dp_epsilon 100 --dp_type 'rdp' --lr 0.01
python $CODE_PATH $DATASET --id 12 --learning_algorithm 'central' --use_dp --local_epochs 20  --target_model='nn' --dp_epsilon 1000 --dp_type 'rdp' --lr 0.01

python $CODE_PATH $DATASET --id 10 --learning_algorithm 'central' --local_epochs 20  --target_model='nn' --lr 0.001
python $CODE_PATH $DATASET --id 11 --learning_algorithm 'central' --use_dp --local_epochs 20  --target_model='nn' --dp_epsilon 10 --dp_type 'rdp' --lr 0.001
python $CODE_PATH $DATASET --id 12 --learning_algorithm 'central' --use_dp --local_epochs 20  --target_model='nn' --dp_epsilon 100 --dp_type 'rdp' --lr 0.001
python $CODE_PATH $DATASET --id 12 --learning_algorithm 'central' --use_dp --local_epochs 20  --target_model='nn' --dp_epsilon 1000 --dp_type 'rdp' --lr 0.001


# python $CODE_PATH $DATASET --learning_algorithm 'central' --use_dp  --local_epochs 10  --target_model='nn' --dp_epsilon 10 --dp_type 'dp' --lr 0.001
# python $CODE_PATH $DATASET --learning_algorithm 'central' --local_epochs 10  --target_model='nn' 

python $CODE_PATH $DATASET --learning_algorithm 'fedakd' --rounds 20 --local_epochs 1  --target_model='nn' --dp_epsilon 1  --lr 0.01
python $CODE_PATH $DATASET --learning_algorithm 'fedakd' --rounds 20 --local_epochs 1  --target_model='nn' --dp_epsilon 1  --lr 0.001

python $CODE_PATH $DATASET --learning_algorithm 'fedakd' --use_dp --rounds 20 --local_epochs 1  --target_model='nn' --dp_epsilon 1 --dp_type 'dp' --lr 0.01
python $CODE_PATH $DATASET --learning_algorithm 'fedakd' --use_dp --rounds 20 --local_epochs 1  --target_model='nn' --dp_epsilon 10 --dp_type 'dp' --lr 0.01
python $CODE_PATH $DATASET --learning_algorithm 'fedakd' --use_dp --rounds 20 --local_epochs 1  --target_model='nn' --dp_epsilon 100 --dp_type 'dp' --lr 0.01
python $CODE_PATH $DATASET --learning_algorithm 'fedakd' --use_dp --rounds 20 --local_epochs 1  --target_model='nn' --dp_epsilon 1000 --dp_type 'dp' --lr 0.01

python $CODE_PATH $DATASET --learning_algorithm 'fedakd' --use_dp --rounds 20 --local_epochs 1  --target_model='nn' --dp_epsilon 1 --dp_type 'rdp' --lr 0.01
python $CODE_PATH $DATASET --learning_algorithm 'fedakd' --use_dp --rounds 20 --local_epochs 1  --target_model='nn' --dp_epsilon 10 --dp_type 'rdp' --lr 0.01
python $CODE_PATH $DATASET --learning_algorithm 'fedakd' --use_dp --rounds 20 --local_epochs 1  --target_model='nn' --dp_epsilon 100 --dp_type 'rdp' --lr 0.01
python $CODE_PATH $DATASET --learning_algorithm 'fedakd' --use_dp --rounds 20 --local_epochs 1  --target_model='nn' --dp_epsilon 1000 --dp_type 'rdp' --lr 0.01
# python $CODE_PATH $DATASET --learning_algorithm 'fedakd' --use_dp --rounds 100 --local_epochs 1  --target_model='nn' --dp_epsilon 100 --dp_type 'rdp' --lr 0.01
# python $CODE_PATH $DATASET --learning_algorithm 'fedakd' --use_dp --rounds 100 --local_epochs 1  --target_model='nn' --dp_epsilon 100 --dp_type 'dp' --lr 0.01
# python $CODE_PATH $DATASET --learning_algorithm 'fedakd' --rounds 20 --local_epochs 1  --target_model='nn' 

