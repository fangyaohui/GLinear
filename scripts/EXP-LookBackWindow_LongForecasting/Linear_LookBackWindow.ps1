# 保存为 run_experiments.ps1
# 执行方式：右键选择"使用PowerShell运行"，或在CMD中执行：powershell -ExecutionPolicy Bypass -File run_experiments.ps1

# 安全设置：遇到错误停止执行
$ErrorActionPreference = "Stop"

# 日志目录设置
$log_dir = "logs\LookBackWindow"
New-Item -ItemType Directory -Force -Path $log_dir | Out-Null

# 数据集参数配置（使用哈希表替代关联数组）
$dataset_params = @{
    "electricity" = "321 custom 16"
    "ETTh1"      = "7 ETTh1 8"
    "traffic"    = "862 custom 16"
    "weather"    = "21 custom 16"
}

# 主循环结构
foreach ($model_name in @("NLinear", "DLinear", "RLinear", "GLinear")) {
    foreach ($lr in @(0.001)) {
        foreach ($seq_len in @(48, 72, 96, 120, 144, 168, 192, 336, 504, 672, 720)) {
            foreach ($pred_len in @(24, 720)) {
                foreach ($dataset in $dataset_params.Keys) {
                    # 参数解析
                    $params = $dataset_params[$dataset] -split ' '
                    $enc_in = $params[0]
                    $data_type = $params[1]
                    $batch_size = $params[2]

                    # 动态生成数据路径
                    $data_file = "$dataset.csv"

                    # 构造日志文件名
                    $log_file = "${model_name}_${dataset}_${seq_len}_${pred_len}_lr_${lr}.log"
                    $log_path = Join-Path -Path $log_dir -ChildPath $log_file

                    # 显示进度信息
                    Write-Host "[$(Get-Date)] Running: $model_name | $dataset | seq:$seq_len pred:$pred_len lr:$lr"

                    # 执行Python命令并记录日志
                    python -u run_longExp.py `
                        --is_training 1 `
                        --root_path ./dataset/ `
                        --data_path "$data_file" `
                        --model_id "${dataset}_${seq_len}_${pred_len}" `
                        --model "$model_name" `
                        --data "$data_type" `
                        --features M `
                        --seq_len $seq_len `
                        --pred_len $pred_len `
                        --enc_in $enc_in `
                        --des 'Exp' `
                        --itr 1 `
                        --batch_size $batch_size `
                        --learning_rate $lr `
                        2>&1 | Out-File -FilePath $log_path -Encoding utf8
                }
            }
        }
    }
}