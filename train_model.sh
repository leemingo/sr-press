python train.py \
--model xgboost \
--trial 0 \
--params_file params.json \
--nb_prev_actions 1 \
--xfns "startlocation" "closest_11_players" \
--yfns "counterpress"


# python train.py \
# --model symbolic_regression \
# --trial 0 \
# --params_file params.json \
# --nb_prev_actions 3 \
# --xfns "startlocation" "closest_11_players" \
# --yfns "counterpress"

python train.py \
--model soccermap \
--trial 0 \
--params_file params.json \
--nb_prev_actions 1 \
--xfns "startlocation" "freeze_frame_360" \
--yfns "counterpress"
