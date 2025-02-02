python train.py \
--model soccermap \
--trial 0 \
--params_file params.json \
--nb_prev_actions 3 \
--xfns "startlocation" "endlocation" "actiontype_onehot" "freeze_frame_360" "result" \
--yfns "possession_change_by_3_seconds_and_3m_distance"
