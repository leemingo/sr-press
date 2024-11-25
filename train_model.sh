python train.py \
--model xgboost \
--trial 806 \
--params_file params.json \
--nb_prev_actions 3 \
--xfns "startlocation" "closest_11_players" "actiontype_onehot" "result_onehot" "bodypart_onehot" "startpolar" "endpolar" "angle" "time_delta" \
--yfns "posession_change"


# python train.py \
# --model symbolic_regression \
# --trial 0 \
# --params_file params.json \
# --nb_prev_actions 3 \
# --xfns "startlocation" "closest_11_players" \
# --yfns "counterpress"

# python train.py \
# --model soccermap \
# --trial 0 \
# --params_file params.json \
# --nb_prev_actions 1 \
# --xfns "startlocation" "freeze_frame_360" \
# --yfns "counterpress"
