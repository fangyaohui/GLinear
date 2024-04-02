
for model_name in NLinear DLinear RLinear GLinear #DNGLinear 
do 
for lr in 0.001                           #0.05 0.001 0.0001 
do
for seq_len in 336 
do
for pred_len in 12 24 48 96 192 336 720 
do
   python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path electricity.csv \
    --model_id Electricity_$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len  \
    --enc_in 321 \
    --des 'Exp' \
    --itr 1 --batch_size 16  --learning_rate $lr >logs/LongForecasting/$model_name'_'electricity_$seq_len'_'$pred_len'_lr_'$lr.log

  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_$seq_len'_'$pred_len \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len  \
    --enc_in 7 \
    --des 'Exp' \
    --itr 1 --batch_size 8  --learning_rate $lr >logs/LongForecasting/$model_name'_'ETTh1_$seq_len'_'$pred_len'_lr_'$lr.log


  python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path traffic.csv \
  --model_id traffic_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len  \
  --enc_in 862 \
  --des 'Exp' \
  --itr 1 --batch_size 16  --learning_rate $lr >logs/LongForecasting/$model_name'_'traffic_$seq_len'_'$pred_len'_lr_'$lr.log

  python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path weather.csv \
  --model_id weather_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len  \
  --enc_in 21 \
  --des 'Exp' \
  --itr 1 --batch_size 16  --learning_rate $lr >logs/LongForecasting/$model_name'_'weather_$seq_len'_'$pred_len'_lr_'$lr.log

  
done
done
done
done

