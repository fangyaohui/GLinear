if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LookBackWindow" ]; then
    mkdir ./logs/LookBackWindow
fi

for model_name in Autoformer 
do 
for seq_len in 24 48 72 96 120 144 168 192 336 504 672 720 
do
for pred_len in 24 720
do

  python -u run_longExp.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path electricity.csv \
      --model_id electricity_96_$pred_len \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 321 \
      --dec_in 321 \
      --c_out 321 \
      --des 'Exp' \
      --itr 1 >logs/LookBackWindow/$model_name'_electricity'_$seq_len'_'$pred_len.log

  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path traffic.csv \
    --model_id traffic_96_$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --des 'Exp' \
    --itr 1 \
    --train_epochs 3 >logs/LookBackWindow/$model_name'_traffic'_$seq_len'_'$pred_len.log

  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path weather.csv \
    --model_id weather_96_$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --itr 1 \
    --train_epochs 2 >logs/LookBackWindow/$model_name'_weather'_$seq_len'_'$pred_len.log

  python -u run_longExp.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ETTh1.csv \
      --model_id ETTh1_96_$pred_len \
      --model $model_name \
      --data ETTh1 \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --itr 1  >logs/LookBackWindow/$model_name'_Etth1'_$seq_len'_'$pred_len.log
  
  
done
done
done
