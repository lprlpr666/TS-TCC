# python main.py --experiment_description exp_pretrainall1dseed34 --run_description run_1 --seed 123 --training_mode self_supervised --selected_dataset SEED
# python main.py --experiment_description exp_pretrainall1dseed3 --run_description run_3 --seed 1 --training_mode fine_tune --selected_dataset SEED --id 1 --scale_channel T --ratio 2
for j in {1..5}
do
    for i in {1..15}
    do
        # python main.py --experiment_description exp_pretrainall1dseed3 --run_description run_2 --seed $j --training_mode supervised --selected_dataset SEED --id $i --scale_channel T --ratio 1
        # python main.py --experiment_description exp_pretrainall1dseed3 --run_description run_2 --seed $j --training_mode supervised --selected_dataset SEED --id $i --scale_channel T --ratio 2
        python main.py --experiment_description exp_pretrainall1dseed3 --run_description run_2 --seed $j --training_mode supervised --selected_dataset SEED --id $i --scale_channel T --ratio 3
    done
done
# python main.py --experiment_description exp_pretrainall1dseed3 --run_description run_3 --seed 123 --training_mode self_supervised --selected_dataset SEED --id 0 --scale_channel F


# exp1 moco test

