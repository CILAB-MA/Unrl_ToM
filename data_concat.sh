folder_name=$1
env_type=$2
data_id=$3
python preprocessing/concat_epi_preprocessing.py --proc_dir /data/$folder_name/preprocess/$env_type -n $data_id  -p 25 -et $env_type
python preprocessing/final_preprocessing.py --proc_dir /data/$folder_name/preprocess/$env_type -n $data_id