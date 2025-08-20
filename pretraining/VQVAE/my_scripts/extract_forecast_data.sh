random_seed=2021
root_path_name=E:/vscode/TOTEM/
data_path_name=forecasting/init_data/electricity.csv
model_id_name=electricity
data_name=custom
seq_len=96
gpu=0
trained_vqvae_model_path=E:/vscode/TOTEM/forecasting/trained_model/generatlist_vqvae/
for pred_len in 96 # 192 336 720
do
python -u forecasting/extract_forecasting_data.py \
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
  --save_path "forecasting/data/all_vqvae_extracted/electricity/Tin"$seq_len"_Tout"$pred_len"/"\
  --trained_vqvae_model_path $trained_vqvae_model_path\
  --compression_factor 4 \
  --classifiy_or_forecast "forecast"
done

seq_len=96
random_seed=2021
root_path_name=E:/vscode/TOTEM/
data_path_name=forecasting/init_data/ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1
gpu=0
trained_vqvae_model_path=E:/vscode/TOTEM/forecasting/trained_model/generatlist_vqvae/
for pred_len in 96 # 192 336 720
do
python -u forecasting/extract_forecasting_data.py \
  --random_seed $random_seed \
  --data $data_name \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --label_len 0 \
  --enc_in 7\
  --gpu $gpu\
  --save_path "forecasting/data/all_vqvae_extracted/ETTh1/Tin"$seq_len"_Tout"$pred_len"/"\
  --trained_vqvae_model_path $trained_vqvae_model_path\
  --compression_factor 4 \
  --classifiy_or_forecast "forecast"
done

seq_len=96
random_seed=2021
root_path_name=E:/vscode/TOTEM/
data_path_name=forecasting/init_data/ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2
gpu=0
trained_vqvae_model_path=E:/vscode/TOTEM/forecasting/trained_model/generatlist_vqvae/
for pred_len in 96 # 192 336 720
do
python -u forecasting/extract_forecasting_data.py \
  --random_seed $random_seed \
  --data $data_name \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --label_len 0 \
  --enc_in 7\
  --gpu $gpu\
  --save_path "forecasting/data/all_vqvae_extracted/ETTh2/Tin"$seq_len"_Tout"$pred_len"/"\
  --trained_vqvae_model_path $trained_vqvae_model_path\
  --compression_factor 4 \
  --classifiy_or_forecast "forecast"
done

random_seed=2021
root_path_name=E:/vscode/TOTEM/
data_path_name=forecasting/init_data/ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1
seq_len=96
gpu=0
trained_vqvae_model_path=E:/vscode/TOTEM/forecasting/trained_model/generatlist_vqvae/
for pred_len in 96 # 192 336 720
do
python -u forecasting/extract_forecasting_data.py \
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
  --save_path "forecasting/data/all_vqvae_extracted/ETTm1/Tin"$seq_len"_Tout"$pred_len"/"\
  --trained_vqvae_model_path $trained_vqvae_model_path\
  --compression_factor 4 \
  --classifiy_or_forecast "forecast"
done

random_seed=2021
root_path_name=E:/vscode/TOTEM/
data_path_name=forecasting/init_data/ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2
seq_len=96
gpu=0
trained_vqvae_model_path=E:/vscode/TOTEM/forecasting/trained_model/generatlist_vqvae/
for pred_len in 96 # 192 336 720
do
python -u forecasting/extract_forecasting_data.py \
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
  --save_path "forecasting/data/all_vqvae_extracted/ETTm2/Tin"$seq_len"_Tout"$pred_len"/"\
  --trained_vqvae_model_path $trained_vqvae_model_path\
  --compression_factor 4 \
  --classifiy_or_forecast "forecast"
done

random_seed=2021
root_path_name=E:/vscode/TOTEM/
data_path_name=forecasting/init_data/traffic.csv
model_id_name=traffic
data_name=custom
seq_len=96
gpu=0
trained_vqvae_model_path=E:/vscode/TOTEM/forecasting/trained_model/generatlist_vqvae/
for pred_len in 96 # 192 336 720
do
python -u forecasting/extract_forecasting_data.py \
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
  --save_path "forecasting/data/all_vqvae_extracted/traffic/Tin"$seq_len"_Tout"$pred_len"/"\
  --trained_vqvae_model_path $trained_vqvae_model_path\
  --compression_factor 4 \
  --classifiy_or_forecast "forecast"
done

gpu=1
random_seed=2021
root_path_name=E:/vscode/TOTEM/
data_path_name=forecasting/init_data/weather.csv
model_id_name=weather
data_name=custom
seq_len=96
gpu=0
trained_vqvae_model_path=E:/vscode/TOTEM/forecasting/trained_model/generatlist_vqvae/
for pred_len in 96 # 192 336 720
do
python -u forecasting/extract_forecasting_data.py \
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
  --save_path "forecasting/data/all_vqvae_extracted/weather/Tin"$seq_len"_Tout"$pred_len"/"\
  --trained_vqvae_model_path $trained_vqvae_model_path\
  --compression_factor 4 \
  --classifiy_or_forecast "forecast"
done