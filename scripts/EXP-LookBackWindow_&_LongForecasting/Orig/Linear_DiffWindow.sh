model_name=GLinear

for pred_len in 12 #24 720
do
for seq_len in 336 #48 72 96 120 144 168 192 336 504 672 720
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
    --itr 1 --batch_size 16  --learning_rate 0.001 >logs/LookBackWindow/$model_name'_'electricity_$seq_len'_'$pred_len.log

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
    --itr 1 --batch_size 8 >logs/LookBackWindow/$model_name'_'ETTh1_$seq_len'_'$pred_len.log

  python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path Etth2.csv \
  --model_id Etth2_$seq_len'_'$pred_len \
  --model $model_name \
  --data Etth2 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len  \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.05 >logs/LookBackWindow/$model_name'_'Etth2_$seq_len'_'$pred_len.log



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
  --itr 1 --batch_size 16 --learning_rate 0.05 >logs/LookBackWindow/$model_name'_'traffic_$seq_len'_'$pred_len.log

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
  --itr 1 --batch_size 16 >logs/LookBackWindow/$model_name'_'weather_$seq_len'_'$pred_len.log
done
done

