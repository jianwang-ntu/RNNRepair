# >>>> 1. generate flip data and corresponding models;
CUDA_VISIBLE_DEVICES=0
no_flip_mode=0
flip_mode=2
flipfirst=1
flipsecond=7
epoch=30
ratio=0.3
start_seed=1000
end_seed=1010
save_id=0

python dataclean_bin.py -path "./dataflip_"$flipfirst"_"$flipsecond"_"$ratio -flip $flip_mode -flipfirst $flipfirst -flipsecond $flipsecond -epoch $epoch -ratio $ratio -start_seed $start_seed -end_seed $end_seed
# >>>> 2. train sgd models;
# python sgd_train.py $start_seed $end_seed $no_flip_mode $ratio $flipfirst $flipsecond
python sgd_train.py $start_seed $end_seed $flip_mode $ratio $flipfirst $flipsecond

# >>>> 3. calculate influence scores with sgd and icml methods.
python sgd_infl.py sst_gru_sgd sgd $epoch $start_seed $end_seed $save_id $flip_mode $ratio $flipfirst $flipsecond
python sgd_infl.py sst_gru_sgd icml $epoch $start_seed $end_seed $save_id $flip_mode $ratio $flipfirst $flipsecond

# >>>> 4. retrain
python retrain.py -epoch $epoch -flipfirst $flipfirst -flipsecond $flipsecond -flip $flip_mode -ratio $ratio -start_seed $start_seed -end_seed $end_seed

# >>>> 5. plot results
python plot_results.py $flipfirst $flipsecond $flip_mode $ratio $epoch $start_seed $end_seed
