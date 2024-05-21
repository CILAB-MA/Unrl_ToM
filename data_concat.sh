device="w8"

for number in $(seq 0 99); do
	python preprocessing/concat_epi_preprocessing.py --proc_dir /data/hoyun_log/preprocess/"$device" -n "$number" -p 1
done;
