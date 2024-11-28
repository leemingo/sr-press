python train.py \
--model xgboost \
--trial 3 \
--params_file params.json \
--nb_prev_actions 3 \
--state_xfns 'actiontype' 'result' 'bodypart' 'time' 'startlocation' 'endlocation' 'startpolar' 'endpolar' 'movement' 'team' 'time_delta' 'space_delta' 'relative_startlocation' 'relative_endlocation' 'angle' 'speed' 'player_possession_time' 'goalscore' '_opponents_in_3m_radius' 'dist_opponent' 'closest_11_players' \
--pressure_state_xfns '_opponents_in_3m_radius' 'dist_opponent' 'closest_11_players' \
--pressure_with_context_xfns 'space_delta' 'relative_defender_angle' 'time_delta' \
--yfns "possession_change_by_4_actions_and_5m_distance"
# --yfns "counterpress"

python train.py \
--model xgboost \
--trial 9 \
--params_file params.json \
--nb_prev_actions 3 \
--state_xfns 'actiontype' 'result' 'bodypart' 'time' 'startlocation' 'endlocation' 'startpolar' 'endpolar' 'movement' 'team' 'time_delta' 'space_delta' 'relative_startlocation' 'relative_endlocation' 'angle' 'player_possession_time' 'goalscore' \
--pressure_with_context_xfns 'space_delta' 'relative_defender_angle' 'time_delta' \
--yfns "possession_change_by_4_actions_and_5m_distance"


python train.py \
--model symbolic_regression \
--trial 0 \
--params_file params.json \
--nb_prev_actions 3 \
--xfns "startlocation" "closest_11_players" \
--yfns "counterpress"

python train.py \
--model soccermap \
--trial 4 \
--params_file params.json \
--nb_prev_actions 3 \
--xfns "startlocation" "freeze_frame_360" \
--yfns "possession_change_by_4_actions"

python train.py \
--model baseline \
--trial 1 \
--params_file params.json \
--nb_prev_actions 1 \
--xfns "startlocation" "freeze_frame_360" \
--yfns "counterpress"
