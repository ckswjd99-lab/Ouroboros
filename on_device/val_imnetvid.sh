MODELS=("vitdet-b" "vitdet-l" "swin-b" "swin-l")
# METHODS=("ours" "evit" "maskvd" "stgt")
METHODS=("evit" "maskvd" "stgt")
# FRAME_RATES=(1 6 30 100)
FRAME_RATES=(30)
TOPKS=(128 256 512 1024)

for model in "${MODELS[@]}"; do
    for method in "${METHODS[@]}"; do
        for frame_rate in "${FRAME_RATES[@]}"; do
            for topk in "${TOPKS[@]}"; do
                python3 evaluate_imvid.py --model "$model" --method "$method" --frame-rates "$frame_rate" --dmap-type "topk" --dirty-topk "$topk"
            done
        done
    done
done
