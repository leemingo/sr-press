python train.py \
--model xgboost \
--trial 0 \
--params_file params.json \
--nb_prev_actions 3 \
--xfns "startlocation" "closest_11_players" \
--yfns "possession_change_by_5_seconds"

python train.py \
--model soccermap \
--trial 1 \
--params_file params.json \
--nb_prev_actions 3 \
--xfns "startlocation" "freeze_frame_360" \
--yfns "possession_change_by_5_seconds"
