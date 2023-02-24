set -xe

NHOSTS=${1:-$(nvidia-smi --list-gpus | wc -l)}

for id in $( seq 1 $((NHOSTS - 1)) );
do
	python mpmd_runner.py \
		--num_processes ${NHOSTS} \
		--process_id ${id} &
done

python mpmd_runner.py \
		--num_processes ${NHOSTS} \
		--process_id 0
