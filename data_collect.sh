device="worker03"

for number in $(seq 0 33); do
    python preprocessing/log_collector.py --base_dir /data/mtap/raw/"$device" -nt 200 -n "$number" -p 25
    python preprocessing/one_epi_preprocessing.py --base_dir /data/mtap/raw/"$device" --proc_dir /data/mtap/preprocess/"$device" -n "$number" -p 25
    python preprocessing/concat_epi_preprocessing.py --proc_dir /data/mtap/preprocess/"$device" -n "$number" -p 25
done;
