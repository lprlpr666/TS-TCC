for i in {1..3}
do
python main.py --experiment_description SEED4_kold --run_description run_1 --seed 123 --training_mode self_supervised --selected_dataset SEED4 --scale_channel F --id 0 --encoder conv1d --file_seed $i
done

for i in {1..3}
do
python main.py --experiment_description SEED_kold --run_description run_1 --seed 123 --training_mode self_supervised --selected_dataset SEED --scale_channel F --id 0 --encoder conv1d --file_seed $i
done