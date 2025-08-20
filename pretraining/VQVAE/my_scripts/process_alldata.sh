#ETTh1
seq_len=96
root_path_name=E:/vscode/TOTEM/
data_path_name=forecasting/init_data/ETTh1.csv
data_name=ETTh1
random_seed=2021
pred_len=96
gpu=0

python -u forecasting/save_revin_data.py \
  --random_seed $random_seed \
  --data $data_name \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --label_len 0 \
  --enc_in 7 \
  --gpu $gpu\
  --save_path "forecasting/data/ETTh1"

# ETTh2
seq_len=96
root_path_name=E:/vscode/TOTEM/
data_path_name=forecasting/init_data/ETTh2.csv
data_name=ETTh2
random_seed=2021
pred_len=96
gpu=0

python -u forecasting/save_revin_data.py \
  --random_seed $random_seed \
  --data $data_name \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --label_len 0 \
  --enc_in 7 \
  --gpu $gpu\
  --save_path "forecasting/data/ETTh2"


# ETTm1
seq_len=96
root_path_name=E:/vscode/TOTEM/
data_path_name=forecasting/init_data/ETTm1.csv
data_name=ETTm1
random_seed=2021
pred_len=96
gpu=0

python -u forecasting/save_revin_data.py \
  --random_seed $random_seed \
  --data $data_name \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --label_len 0 \
  --enc_in 7 \
  --gpu $gpu\
  --save_path "forecasting/data/ETTm1"


# ETTm2
seq_len=96
root_path_name=E:/vscode/TOTEM/
data_path_name=forecasting/init_data/ETTm2.csv
data_name=ETTm2
random_seed=2021
pred_len=96
gpu=0

python -u forecasting/save_revin_data.py \
  --random_seed $random_seed \
  --data $data_name \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --label_len 0 \
  --enc_in 7 \
  --gpu $gpu\
  --save_path "forecasting/data/ETTm2"


# traffic
seq_len=96
root_path_name=E:/vscode/TOTEM/
data_path_name=forecasting/init_data/traffic.csv
data_name=custom
random_seed=2021
pred_len=96
gpu=0

python -u forecasting/save_revin_data.py\
  --random_seed $random_seed \
  --data $data_name \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --label_len 0 \
  --enc_in 862 \
  --gpu $gpu\
  --save_path "forecasting/data/traffic"


# weather
seq_len=96
root_path_name=E:/vscode/TOTEM/
data_path_name=forecasting/init_data/weather.csv
data_name=custom
random_seed=2021
pred_len=96
gpu=0

python -u forecasting/save_revin_data.py \
  --random_seed $random_seed \
  --data $data_name \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --label_len 0 \
  --enc_in 21 \
  --gpu $gpu\
  --save_path "forecasting/data/weather"


# electricity
seq_len=96
root_path_name=E:/vscode/TOTEM/
data_path_name=forecasting/init_data/electricity.csv
data_name=custom
random_seed=2021
pred_len=96
gpu=0

python -u forecasting/save_revin_data.py \
  --random_seed $random_seed \
  --data $data_name \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --label_len 0 \
  --enc_in 321 \
  --gpu $gpu\
  --save_path "forecasting/data/electricity"