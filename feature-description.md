| Feature Group         | Feature Name                | Description                                                                                     |
|-----------------------|-----------------------------|-------------------------------------------------------------------------------------------------|
| Action Type           | type_pass                   | One-hot encoded for each action type (e.g., pass, cross, throw-in, etc.).                       |
|                       | type_cross                  |                                                                                                 |
|                       | type_throw_in               |                                                                                                 |
|                       | ...                         |                                                                                                 |
| Result                | result_success              | One-hot encoded result of each action (e.g., success, failure).                                 |
|                       | result_failure              |                                                                                                 |
|                       | ...                         |                                                                                                 |
| Body Part             | bodypart_head               | One-hot encoded for each body part used (e.g., foot, head).                                     |
|                       | bodypart_foot               |                                                                                                 |
| Time                  | period_id                   | Period ID of the game.                                                                          |
|                       | time_seconds                | Time in seconds since the start of the period.                                                  |
|                       | time_seconds_overall        | Time in seconds since the start of the game.                                                    |
| Location              | start_x                     | X-coordinate where the action started.                                                          |
|                       | start_y                     | Y-coordinate where the action started.                                                          |
|                       | end_x                       | X-coordinate where the action ended.                                                            |
|                       | end_y                       | Y-coordinate where the action ended.                                                            |
| Relative Location     | start_dist_sideline         | Distance from the sideline where the action started.                                            |
|                       | start_dist_goalline         | Distance from the goalline where the action started.                                            |
|                       | end_dist_sideline           | Distance from the sideline where the action ended.                                              |
|                       | end_dist_goalline           | Distance from the goalline where the action ended.                                              |
| Polar Coordinates     | start_dist_to_goal          | Distance to the goal from the action’s start location.                                          |
|                       | start_angle_to_goal         | Angle to the goal from the action’s start location.                                             |
|                       | end_dist_to_goal            | Distance to the goal from the action’s end location.                                            |
|                       | end_angle_to_goal           | Angle to the goal from the action’s end location.                                               |
| Movement              | dx                          | Horizontal distance covered by the action.                                                      |
|                       | dy                          | Vertical distance covered by the action.                                                        |
|                       | movement                    | Total distance covered by the action.                                                           |
| Team                  | team_<n>                    | Whether the possession was with the same team as the previous action (True/False).              |
| Time Delta            | time_delta_<n>              | Time difference in seconds between the last action and the previous actions.                    |
| Space Delta           | dx_a0i, dy_a0i, mov_a0i     | Distance covered between the last and previous actions (for each previous action).              |
| Goal Score            | goalscore_team              | Number of goals scored by the team after the action.                                            |
|                       | goalscore_opponent          | Number of goals scored by the opponent team after the action.                                   |
|                       | goalscore_diff              | Goal difference after the action.                                                               |
| Angle                 | angle                       | Angle between the start and end location of an action.                                          |
| Pressure              | under_pressure              | Whether the action was performed under pressure.                                                |
| Packing Rate          | packing_rate                | Number of defenders outplayed by a pass.                                                        |
| Ball Height           | ball_height_ground          | One-hot encoded height of the ball (e.g., ground, low, high) during pass-like actions.          |
| Possession Time       | player_possession_time      | Time a player held ball possession before attempting the action.                                |
| Speed                 | speedx_a0i, speedy_a0i      | Ball speed in m/s between the last action and previous actions.                                 |
| Opponents in Path     | nb_opp_in_path              | Number of opponents in the path between start and end locations of a pass.                      |
| Distance to Opponent  | dist_defender_start         | Distance to the nearest defender at the action's start location.                                |
|                       | dist_defender_end           | Distance to the nearest defender at the action's end location.                                  |
|                       | dist_defender_action        | Distance to the nearest defender along the action's path.                                       |
| Defenders in Radius   | nb_defenders_start_3m       | Number of defenders within 3 meters of the action's start location.                             |
|                       | nb_defenders_end_3m         | Number of defenders within 3 meters of the action's end location.                               |
| Closest Players       | teammate_(1-3)_x, teammate_(1-3)_y  | Coordinates and distance of the closest teammates (1-3) to the action’s location.               |
|                       | opponent_(1-3)_x, opponent_(1-3)_y  | Coordinates and distance of the closest opponents (1-3) to the action’s location.               |
