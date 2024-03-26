for i in {1..3}
do
python main_pipeline.py --experiment_description SEED4_kold --run_description run_1 --training_mode fine_tune --selected_dataset SEED --scale_channel T --ratio 0.01 --file_seed $i
python main_pipeline.py --experiment_description SEED4_kold --run_description run_1 --training_mode fine_tune --selected_dataset SEED --scale_channel T --ratio 0.05 --file_seed $i
python main_pipeline.py --experiment_description SEED4_kold --run_description run_1 --training_mode fine_tune --selected_dataset SEED --scale_channel T --ratio 0.1 --file_seed $i
python main_pipeline.py --experiment_description SEED4_kold --run_description run_1 --training_mode fine_tune --selected_dataset SEED --scale_channel T --ratio 1.0 --file_seed $i
done