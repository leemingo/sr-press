"""Provides some utilities widely used by other modules."""
from typing import Dict, List, Sequence, Union
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, Point, box
from scipy.spatial import Voronoi

import express.config as config

def add_names(actions: pd.DataFrame) -> pd.DataFrame:
    """Add the type name(include Pressing), result name and bodypart name to a SPADL(include Pressing) dataframe.

    Parameters
    ----------
    actions : pd.DataFrame
        A SPADL dataframe.

    Returns
    -------
    pd.DataFrame
        The original dataframe with a 'type_name', 'result_name' and
        'bodypart_name' appended.
    """

    return actions.drop(columns=["type_name", "result_name", "bodypart_name"], errors="ignore") \
        .merge(config.actiontypes_df(), how="left") \
        .merge(config.results_df(), how="left") \
        .merge(config.bodyparts_df(), how="left") \
        .set_index(actions.index)

def play_left_to_right(gamestates: List[pd.DataFrame], home_team_id: int) -> List[pd.DataFrame]:
    """Perform all action in the same playing direction.

    This changes the start and end location of each action and the freeze
    frame, such that all actions are performed as if the team plays from left
    to right.

    Parameters
    ----------
    gamestates : GameStates
        The game states of a game.
    home_team_id : int
        The ID of the home team.

    Returns
    -------
    GameStates
        The game states with all actions performed left to right.
    """
    a0 = gamestates[0]
    away_idx = a0.team_id != home_team_id
    for actions in gamestates:
        for col in ["start_x", "end_x"]:
            actions.loc[away_idx, col] = config.field_length - actions[away_idx][col].values
        for col in ["start_y", "end_y"]:
            actions.loc[away_idx, col] = config.field_width - actions[away_idx][col].values

        for idx, action in actions.loc[away_idx].iterrows():
            freeze_frame = action["freeze_frame_360"]

            if freeze_frame is not None:
                freezedf = pd.DataFrame(freeze_frame).fillna(
                    {"teammate": False, "actor": False, "keeper": False}
                )
                freezedf["x"] = config.field_length - freezedf["x"].values
                freezedf["y"] = config.field_width - freezedf["y"].values
                actions.at[idx, "freeze_frame_360"] = freezedf.to_dict(orient="records")

    return gamestates

def calc_voronoi(freeze_frame):
    points = pd.DataFrame()
    points['x'] = freeze_frame.x
    points['y'] = freeze_frame.y
    points = points[
        (points['x'] >= 0) & (points['x'] <= config.field_length) &
        (points['y'] >= 0) & (points['y'] <= config.field_width)
    ]
    points_extended = np.hstack([
        np.vstack([points['x'], points['y']]),
        np.vstack([-points['x'], points['y']]),
        np.vstack([-points['x'] + config.field_length * 2, points['y']]),
        np.vstack([points['x'], -points['y']]),
        np.vstack([points['x'], -points['y'] + config.field_width * 2])
    ]).T

    vor = Voronoi(points_extended)

    return vor

# 골키퍼 위치 보간 함수 정의
# 보간된 골키퍼 선수의 위치를 추가하는 함수
def interpolate_goalkeepers(freeze_frame):
    if not any(freeze_frame["keeper"] & freeze_frame["teammate"]):
        teammate_keeper = pd.DataFrame({
            "teammate": [True],
            "actor": [False],
            "keeper": [True],
            "x": [config.field_length],
            "y": [config.field_width / 2]
        })
        freeze_frame = pd.concat([freeze_frame, teammate_keeper], ignore_index=True)

    if not any(freeze_frame["keeper"] & ~freeze_frame["teammate"]):
        opponent_keeper = pd.DataFrame({
            "teammate": [False],
            "actor": [False],
            "keeper": [True],
            "x": [0],
            "y": [config.field_width / 2]
        })
        freeze_frame = pd.concat([freeze_frame, opponent_keeper], ignore_index=True)
    
    return freeze_frame

# 보로노이 꼭짓점을 활용한 선수 보간 함수 정의
def interpolate_with_voronoi_vertices(freeze_frame, visible_polygon):
    vor = calc_voronoi(freeze_frame)

    teammate_locs = freeze_frame[freeze_frame.teammate].copy()
    opponent_locs = freeze_frame[~freeze_frame.teammate].copy()
    num_teammates = len(teammate_locs)
    num_opponents = len(opponent_locs)

    interpolated_players = []
    # 휴리스틱 기반 접근: visible_area 밖의 Voronoi 다각형의 꼭짓점을 활용하여 선수 보간 수행
    for vertex in vor.vertices:
        if num_teammates == 11 and num_opponents == 11:
            break

        point = Point(vertex[0], vertex[1])
        if not visible_polygon.contains(point) and 0 <= point.x <= config.field_length and 0 <= point.y <= config.field_width:
            # 각 팀의 포착된 선수들과의 거리 계산 후, 가장 가까운 팀으로 보간 수행
            distances_to_team = np.sqrt((teammate_locs['x'] - point.x) ** 2 + (teammate_locs['y'] - point.y) ** 2) 
            distances_to_opponent = np.sqrt((opponent_locs['x'] - point.x) ** 2 + (opponent_locs['y'] - point.y) ** 2) 
            
            interpolate_teammate = (
                (num_teammates < 11 and num_opponents < 11 and distances_to_team.min() < distances_to_opponent.min()) or
                (num_teammates < 11 and num_opponents == 11)
            )
            new_player = pd.DataFrame({
                "teammate": [interpolate_teammate],
                "actor": [False],
                "keeper": [False],
                "x": [point.x],
                "y": [point.y]
            })
            interpolated_players.append(new_player)

            if interpolate_teammate:
                num_teammates += 1
            else:
                num_opponents += 1  

    for player in interpolated_players:
        freeze_frame = pd.concat([freeze_frame, player], ignore_index=True)
    
    return freeze_frame, interpolated_players

# 보로노이 영역의 중심을 활용한 선수 보간 함수 정의
def interpolate_with_voronoi_centroids(freeze_frame, visible_polygon):
    vor = calc_voronoi(freeze_frame)

    teammate_locs = freeze_frame[freeze_frame.teammate].copy()
    opponent_locs = freeze_frame[~freeze_frame.teammate].copy()
    num_teammates = len(teammate_locs)
    num_opponents = len(opponent_locs)

    interpolated_players = []
    field_bounds = box(0, 0, config.field_length, config.field_width)
    region_centroids = []
    for region_idx in vor.point_region:
        # 보로노이 영역의 인덱스를 통해 영역 가져오기
        region = vor.regions[region_idx]
        if not region or -1 in region:
            continue  # 무효 영역인 경우 건너뜀
        
        # 보로노이 영역의 꼭짓점 좌표를 가져와 폴리곤 생성 후 필드 경계와의 교집합 구하기
        polygon_points = [vor.vertices[i] for i in region]
        polygon = Polygon(polygon_points).intersection(field_bounds)

        centroid = polygon.centroid
        # visible_area에 포함되지 않는 중심값만 사용
        if not visible_polygon.contains(centroid):
            region_centroids.append(centroid)

    print("region_centroids: ", region_centroids)
    # 모든 영역의 중심값에 대해 선수와의 거리 계산 및 보간 처리
    for centroid in region_centroids:
        if num_teammates == 11 and num_opponents == 11:
            break
        
        distances_to_team = np.sqrt((teammate_locs['x'] - centroid.x) ** 2 + (teammate_locs['y'] - centroid.y) ** 2)
        distances_to_opponent = np.sqrt((opponent_locs['x'] - centroid.x) ** 2 + (opponent_locs['y'] - centroid.y) ** 2)

        # 팀원과 상대팀 중 더 가까운 쪽으로 보간할지 결정
        interpolate_teammate = (
            (num_teammates < 11 and num_opponents < 11 and distances_to_team.min() < distances_to_opponent.min()) or
            (num_teammates < 11 and num_opponents == 11)
        )

        # 새로운 선수 추가
        new_player = pd.DataFrame({
            "teammate": [interpolate_teammate],
            "actor": [False],
            "keeper": [False],
            "x": [centroid.x],
            "y": [centroid.y]
        })
        interpolated_players.append(new_player)

        print(centroid, num_teammates, num_opponents)
        if interpolate_teammate:
            num_teammates += 1
        else:
            num_opponents += 1

    for player in interpolated_players:
        freeze_frame = pd.concat([freeze_frame, player], ignore_index=True)
    
    return freeze_frame, interpolated_players

# 전체 보간 함수 정의
def interpolate_freeze_frame(freeze_frame, visible_area):
    visible_polygon = Polygon(visible_area)
    
    # 골키퍼 보간
    freeze_frame = interpolate_goalkeepers(freeze_frame)
    print(freeze_frame.shape)
    # 보로노이 꼭짓점 보간
    freeze_frame, interpolated_vertices = interpolate_with_voronoi_vertices(freeze_frame, visible_polygon)
    print(freeze_frame.shape)
    # 보로노이 영역의 중심 보간
    #freeze_frame, interpolated_centroids = interpolate_with_voronoi_centroids(freeze_frame, visible_polygon)
    print(freeze_frame.shape)
    
    return freeze_frame, interpolated_vertices
