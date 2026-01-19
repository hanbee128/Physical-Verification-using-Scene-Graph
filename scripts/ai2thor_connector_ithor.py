#!/usr/bin/env python3
"""
AI2-THOR Connector - ProgPrompt 형식의 PLAN을 실행할 수 있는 클래스

ProgPrompt 형식의 코드를 파싱하고 AI2-THOR에서 실행합니다.
예: GoTo('Apple'), Pickup('Apple'), Put('Apple', 'Fridge') 등
"""

import cv2
import heapq
import math
import numpy as np
import os
import re
import shutil
import threading
import time
from datetime import datetime
from glob import glob
from typing import Dict, List, Optional, Tuple, Any

try:
    from ai2thor.controller import Controller
except ImportError:
    print("⚠️  ai2thor not installed. Please install: pip install ai2thor")

try:
    from utils_aug_env import normalize_object_type
except ImportError:
    # Fallback if import fails
    def normalize_object_type(obj_type: str) -> str:
        """Normalize object type (ButterKnife -> Knife)"""
        if obj_type == "ButterKnife":
            return "Knife"
        return obj_type


def distance_pts(pt1, pt2):
    """두 점 사이의 거리 계산"""
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(pt1, pt2)]))


def closest_node(target_pos, reachable_positions, no_agents, clost_node_location):
    """타겟 위치에 가장 가까운 도달 가능한 위치들을 찾기"""
    distances = []
    for pos in reachable_positions:
        dist = distance_pts(target_pos, pos)
        distances.append((dist, pos))
    
    distances.sort(key=lambda x: x[0])
    
    # 각 agent마다 다른 위치 할당
    selected = []
    for i in range(no_agents):
        idx = (clost_node_location[i] + i) % len(distances)
        selected.append(distances[idx][1])
    
    return selected


class AI2ThorExecutor:
    """ProgPrompt 형식의 PLAN을 AI2-THOR에서 실행하는 클래스"""
    
    def __init__(
        self,
        scene: str = "FloorPlan1",
        agent_id: int = 0,
        headless: bool = False,
        save_images: bool = False,
        image_dir: str = "execution_images",
        save_video: bool = True,
        video_fps: float = 10.0,
        video_dir: str = "execution_videos",
        initial_position: Optional[Tuple[float, float, float]] = None,
        initial_position_strategy: str = "first",  # "first", "random", "center", "custom"
    ):
        """
        Args:
            scene: AI2-THOR scene 이름
            agent_id: Agent ID (기본값: 0)
            headless: 헤드리스 모드 (GUI 없이 실행)
            save_images: 이미지 저장 여부
            image_dir: 이미지 저장 디렉토리
            save_video: 영상 저장 여부 (기본값: True)
            video_fps: 영상 FPS (기본값: 30.0, 실제보다 빠르게 재생)
            video_dir: 영상 저장 디렉토리
            initial_position: 초기 위치 (x, y, z) 튜플. None이면 strategy에 따라 결정
            initial_position_strategy: 초기 위치 선택 전략
                - "first": 첫 번째 도달 가능한 위치 (기본값)
                - "random": 랜덤 위치
                - "center": 씬의 중심에 가까운 위치
                - "custom": initial_position 사용
        """
        self.scene = scene
        self.agent_id = agent_id
        self.headless = headless
        self.save_images = save_images
        self.image_dir = image_dir
        self.save_video = save_video
        self.video_fps = video_fps
        self.video_dir = video_dir
        self.initial_position = initial_position
        self.initial_position_strategy = initial_position_strategy
        
        self.controller = None
        self.action_queue = []
        self.task_over = False
        self.reachable_positions = []
        self.total_exec = 0
        self.success_exec = 0
        self.execution_log = []
        
        # 비디오 저장 관련
        self.video_writer = None
        self.video_frames = []
        self.video_width = 1000
        self.video_height = 1000
        
        # NavMesh 그래프 (A* 경로 탐색용, lazy initialization)
        self.navmesh_graph = None
        
        # 이미지 저장 관련
        if self.save_images:
            self._setup_image_dirs()
        
        # 비디오 저장 디렉토리 설정
        if self.save_video:
            os.makedirs(self.video_dir, exist_ok=True)
    
    def _setup_image_dirs(self):
        """이미지 저장 디렉토리 설정"""
        if os.path.exists(self.image_dir):
            shutil.rmtree(self.image_dir)
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(os.path.join(self.image_dir, "agent"), exist_ok=True)
        os.makedirs(os.path.join(self.image_dir, "top_view"), exist_ok=True)
    
    def _init_video_writer(self):
        """비디오 writer 초기화"""
        if not self.controller:
            return
        
        # 첫 프레임으로 비디오 크기 결정
        try:
            event = self.controller.last_event
            if hasattr(event, 'cv2img') and event.cv2img is not None:
                frame = event.cv2img
                self.video_height, self.video_width = frame.shape[:2]
            elif hasattr(event, 'frame') and event.frame is not None:
                frame = event.frame
                self.video_height, self.video_width = frame.shape[:2]
            elif hasattr(event, 'events') and len(event.events) > self.agent_id:
                agent_event = event.events[self.agent_id]
                if hasattr(agent_event, 'cv2img') and agent_event.cv2img is not None:
                    frame = agent_event.cv2img
                    self.video_height, self.video_width = frame.shape[:2]
                elif hasattr(agent_event, 'frame') and agent_event.frame is not None:
                    frame = agent_event.frame
                    self.video_height, self.video_width = frame.shape[:2]
            else:
                # 기본값 사용
                self.video_width = 1000
                self.video_height = 1000
        except Exception as e:
            print(f"⚠️ Could not determine video size: {e}, using default 1000x1000")
            self.video_width = 1000
            self.video_height = 1000
        
        # 비디오 파일명 생성 (타임스탬프 포함)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"execution_{self.scene}_{timestamp}.mp4"
        video_path = os.path.join(self.video_dir, video_filename)
        
        # VideoWriter 초기화 (MP4V 코덱 사용)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            video_path,
            fourcc,
            self.video_fps,
            (self.video_width, self.video_height)
        )
        
        if not self.video_writer.isOpened():
            print(f"⚠️ Failed to initialize video writer")
            self.video_writer = None
        else:
            print(f"✓ Video writer initialized: {video_path} (FPS: {self.video_fps})")
            self.video_path = video_path
    
    def _capture_frame(self):
        """현재 프레임을 캡처하여 비디오에 추가"""
        if not self.save_video or not self.controller:
            return
        
        try:
            event = self.controller.last_event
            
            # Agent view 프레임 가져오기
            frame = None
            if hasattr(event, 'cv2img') and event.cv2img is not None:
                frame = event.cv2img.copy()
            elif hasattr(event, 'frame') and event.frame is not None:
                frame = event.frame.copy()
            elif hasattr(event, 'events') and len(event.events) > self.agent_id:
                agent_event = event.events[self.agent_id]
                if hasattr(agent_event, 'cv2img') and agent_event.cv2img is not None:
                    frame = agent_event.cv2img.copy()
                elif hasattr(agent_event, 'frame') and agent_event.frame is not None:
                    frame = agent_event.frame.copy()
            
            if frame is not None:
                # 프레임 크기 조정 (비디오 크기에 맞춤)
                if frame.shape[:2] != (self.video_height, self.video_width):
                    frame = cv2.resize(frame, (self.video_width, self.video_height))
                
                # 비디오에 프레임 추가
                if self.video_writer and self.video_writer.isOpened():
                    self.video_writer.write(frame)
                    self.video_frames.append(frame)
        except Exception as e:
            print(f"⚠️ Error capturing frame: {e}")
    
    def _close_video_writer(self):
        """비디오 writer 종료 및 파일 저장"""
        if self.video_writer and self.video_writer.isOpened():
            self.video_writer.release()
            self.video_writer = None
            if hasattr(self, 'video_path'):
                print(f"✓ Video saved: {self.video_path} ({len(self.video_frames)} frames)")
            self.video_frames = []
    
    def initialize(self):
        """AI2-THOR 환경 초기화"""
        self.controller = Controller(height=1000, width=1000, headless=self.headless)
        self.controller.reset(self.scene)
        
        # Agent 초기화
        self.controller.step(dict(
            action='Initialize',
            agentMode="default",
            snapGrid=False,
            gridSize=0.25,
            rotateStepDegrees=20,
            visibilityDistance=1.5,
            fieldOfView=120,
            agentCount=1
        ))
        
        # Top view camera 추가
        event = self.controller.step(action="GetMapViewCameraProperties")
        self.controller.step(action="AddThirdPartyCamera", **event.metadata["actionReturn"])
        
        # 도달 가능한 위치 가져오기 (NavMesh 기반)
        reachable_positions_ = self.controller.step(action="GetReachablePositions").metadata["actionReturn"]
        self.reachable_positions = [(p["x"], p["y"], p["z"]) for p in reachable_positions_]
        
        # NavMesh 그래프 초기화 (A* 경로 탐색용)
        print(f"  Building NavMesh graph from {len(self.reachable_positions)} reachable positions...")
        self.navmesh_graph = self._build_navmesh_graph(max_connection_distance=1.5)
        print(f"  ✓ NavMesh graph built: {len(self.navmesh_graph)} nodes")
        
        # Agent 초기 위치 설정
        init_pos = self._select_initial_position()
        if init_pos:
            self.controller.step(dict(action="Teleport", position=init_pos, agentId=self.agent_id))
            print(f"  ✓ Agent initialized at position: ({init_pos['x']:.2f}, {init_pos['y']:.2f}, {init_pos['z']:.2f})")
        else:
            print(f"  ⚠ No valid initial position found, agent will remain at default position")
        
        # Agent 시야 조정
        self.controller.step(action="LookDown", degrees=35, agentId=self.agent_id)
        
        # 비디오 writer 초기화
        if self.save_video:
            self._init_video_writer()
        
        print(f"✓ AI2-THOR initialized: {self.scene}")
    
    def _select_initial_position(self) -> Optional[Dict[str, float]]:
        """
        초기 위치 선택 (사용자 지정 또는 전략 기반)
        
        Returns:
            초기 위치 딕셔너리 {"x": float, "y": float, "z": float} 또는 None
        """
        if not self.reachable_positions:
            return None
        
        # 사용자가 직접 위치를 지정한 경우
        if self.initial_position_strategy == "custom" and self.initial_position:
            # 지정된 위치가 도달 가능한 위치 중에 있는지 확인
            x, y, z = self.initial_position
            # 가장 가까운 도달 가능한 위치 찾기
            closest = min(
                self.reachable_positions,
                key=lambda p: distance_pts([x, y, z], list(p))
            )
            dist = distance_pts([x, y, z], list(closest))
            if dist < 0.5:  # 0.5m 이내면 사용
                return {"x": closest[0], "y": closest[1], "z": closest[2]}
            else:
                print(f"  ⚠ Custom position ({x:.2f}, {y:.2f}, {z:.2f}) is not reachable, using closest: {dist:.2f}m away")
                return {"x": closest[0], "y": closest[1], "z": closest[2]}
        
        # 전략 기반 선택
        if self.initial_position_strategy == "random":
            import random
            pos = random.choice(self.reachable_positions)
            return {"x": pos[0], "y": pos[1], "z": pos[2]}
        
        elif self.initial_position_strategy == "center":
            # 모든 도달 가능한 위치의 중심 계산
            if not self.reachable_positions:
                return None
            
            center_x = sum(p[0] for p in self.reachable_positions) / len(self.reachable_positions)
            center_y = sum(p[1] for p in self.reachable_positions) / len(self.reachable_positions)
            center_z = sum(p[2] for p in self.reachable_positions) / len(self.reachable_positions)
            center = [center_x, center_y, center_z]
            
            # 중심에 가장 가까운 위치 선택
            closest = min(
                self.reachable_positions,
                key=lambda p: distance_pts(center, list(p))
            )
            return {"x": closest[0], "y": closest[1], "z": closest[2]}
        
        else:  # "first" (기본값)
            pos = self.reachable_positions[0]
            return {"x": pos[0], "y": pos[1], "z": pos[2]}
    
    def _find_object_id(self, object_name: str) -> Optional[str]:
        """
        객체 이름, objectId, 또는 좌표로 objectId 찾기
        
        Args:
            object_name: 
                - 객체 이름 (예: "Cabinet")
                - objectId (예: "Cabinet_123")
                - 좌표 포함 형식 (예: "Cabinet|+00.72|+02.02|-02.46")
            
        Returns:
            objectId 또는 None
        """
        all_objects = self.controller.last_event.metadata.get("objects", [])
        objs = [obj["objectId"] for obj in all_objects]
        
        # objectId가 직접 주어진 경우 (정확한 매칭)
        if object_name in objs:
            return object_name
        
        # 좌표 포함 형식 파싱 (예: "Cabinet|+00.72|+02.02|-02.46" 또는 "Cabinet|+0.72|+2.02|-2.46")
        if "|" in object_name:
            parts = object_name.split("|")
            if len(parts) >= 4:  # 타입|좌표1|좌표2|좌표3
                obj_type = parts[0]
                try:
                    # 좌표 문자열 파싱 (+00.72 형식 또는 +0.72 형식 모두 지원)
                    x_str = parts[1].strip()
                    y_str = parts[2].strip()
                    z_str = parts[3].strip()
                    target_x = float(x_str)
                    target_y = float(y_str)
                    target_z = float(z_str)
                    
                    # 좌표로 가장 가까운 객체 찾기
                    closest_obj = None
                    min_distance = float('inf')
                    
                    for obj in all_objects:
                        obj_type_normalized = normalize_object_type(obj.get("objectType", ""))
                        if obj_type_normalized.lower() != obj_type.lower():
                            continue
                        
                        # 객체의 중심 위치 가져오기
                        bbox = obj.get("axisAlignedBoundingBox", {})
                        center = bbox.get("center", {})
                        if not center:
                            continue
                        
                        obj_x = center.get("x", 0)
                        obj_y = center.get("y", 0)
                        obj_z = center.get("z", 0)
                        
                        # 유클리드 거리 계산
                        distance = math.sqrt(
                            (obj_x - target_x)**2 + 
                            (obj_y - target_y)**2 + 
                            (obj_z - target_z)**2
                        )
                        
                        if distance < min_distance:
                            min_distance = distance
                            closest_obj = obj
                    
                    if closest_obj and min_distance < 0.5:  # 0.5m 이내면 매칭
                        print(f"  ✓ Found {obj_type} at coordinates ({target_x:.2f}, {target_y:.2f}, {target_z:.2f})")
                        return closest_obj.get("objectId")
                    elif closest_obj:
                        print(f"  ⚠ Closest {obj_type} found at distance {min_distance:.2f}m from target coordinates")
                        return closest_obj.get("objectId")
                except ValueError:
                    # 좌표 파싱 실패 시 objectId로 처리 (예: "Cabinet|Cabinet_123")
                    object_id_candidate = parts[-1]  # 마지막 부분이 objectId일 수 있음
                    if object_id_candidate in objs:
                        return object_id_candidate
                    # objectId가 아니면 타입 이름만 사용
                    object_name = parts[0]
        
        # 정확한 매칭 먼저 시도
        for obj_id in objs:
            if object_name.lower() in obj_id.lower() or obj_id.lower().startswith(object_name.lower()):
                return obj_id
        
        # 정규식 매칭
        for obj_id in objs:
            if re.match(object_name, obj_id, re.IGNORECASE):
                return obj_id
        
        return None
    
    def _get_object_center(self, object_id: str) -> Optional[Dict[str, float]]:
        """객체의 중심 위치 가져오기"""
        for obj in self.controller.last_event.metadata["objects"]:
            if obj["objectId"] == object_id:
                return obj.get("axisAlignedBoundingBox", {}).get("center")
        return None
    
    def _find_closest_reachable_position(self, target_pos: List[float], max_neighbors: int = 10) -> List[Tuple[float, float, float]]:
        """
        목표 위치에 가장 가까운 도달 가능한 위치들을 반환 (A* 경로 탐색용)
        
        Args:
            target_pos: 목표 위치 [x, y, z]
            max_neighbors: 반환할 최대 후보 개수
            
        Returns:
            가장 가까운 도달 가능한 위치들의 리스트
        """
        if not self.reachable_positions:
            return []
        
        # 거리순으로 정렬
        sorted_positions = sorted(
            self.reachable_positions,
            key=lambda p: distance_pts(target_pos, p)
        )
        
        return sorted_positions[:max_neighbors]
    
    def _build_navmesh_graph(self, max_connection_distance: float = 1.5) -> Dict[Tuple[float, float, float], List[Tuple[Tuple[float, float, float], float]]]:
        """
        NavMesh의 이동 가능한 위치들을 노드로 하는 그래프 생성
        각 노드는 일정 거리 내의 다른 노드들과 연결됨
        
        Args:
            max_connection_distance: 두 노드를 연결할 최대 거리 (미터)
            
        Returns:
            그래프 딕셔너리: {노드: [(연결된_노드, 거리), ...]}
        """
        graph = {}
        
        for pos1 in self.reachable_positions:
            neighbors = []
            for pos2 in self.reachable_positions:
                if pos1 == pos2:
                    continue
                dist = distance_pts(pos1, pos2)
                if dist <= max_connection_distance:
                    neighbors.append((pos2, dist))
            graph[tuple(pos1)] = neighbors
        
        return graph
    
    def _astar_pathfinding(
        self, 
        start_pos: Tuple[float, float, float], 
        goal_pos: Tuple[float, float, float],
        graph: Dict[Tuple[float, float, float], List[Tuple[Tuple[float, float, float], float]]],
        max_search_nodes: int = 500
    ) -> Optional[List[Tuple[float, float, float]]]:
        """
        A* 알고리즘을 사용한 최단 경로 탐색
        
        Args:
            start_pos: 시작 위치 (x, y, z)
            goal_pos: 목표 위치 (x, y, z)
            graph: NavMesh 그래프
            max_search_nodes: 최대 탐색 노드 수
            
        Returns:
            경로 (위치 리스트) 또는 None (경로 없음)
        """
        start = tuple(start_pos)
        goal = tuple(goal_pos)
        
        # 시작점과 목표점이 그래프에 없으면 가장 가까운 노드 찾기
        if start not in graph:
            closest_start = min(
                graph.keys(),
                key=lambda p: distance_pts(list(start), list(p))
            )
            start = closest_start
        
        if goal not in graph:
            closest_goal = min(
                graph.keys(),
                key=lambda p: distance_pts(list(goal), list(p))
            )
            goal = closest_goal
        
        # A* 알고리즘
        open_set = [(0, start)]  # (f_score, node)
        came_from = {}
        g_score = {start: 0}  # 시작점에서 각 노드까지의 실제 거리
        f_score = {start: distance_pts(list(start), list(goal))}  # 휴리스틱 (예상 총 거리)
        visited = set()
        
        search_count = 0
        
        while open_set and search_count < max_search_nodes:
            current_f, current = heapq.heappop(open_set)
            
            if current in visited:
                continue
            
            visited.add(current)
            search_count += 1
            
            # 목표 도달
            if distance_pts(list(current), list(goal)) < 0.5:
                # 경로 재구성
                path = [list(goal)]
                node = current
                while node in came_from:
                    path.append(list(node))
                    node = came_from[node]
                path.append(list(start))
                path.reverse()
                return path
            
            # 인접 노드 탐색
            if current not in graph:
                continue
                
            for neighbor, edge_cost in graph[current]:
                if neighbor in visited:
                    continue
                
                tentative_g = g_score[current] + edge_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    h_score = distance_pts(list(neighbor), list(goal))  # 휴리스틱
                    f_score[neighbor] = tentative_g + h_score
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # 경로를 찾지 못함
        return None
    
    def goto_object(self, object_name: str, target_distance: float = 1.0, max_steps: int = 50, use_astar: bool = True) -> bool:
        """
        객체로 이동 (A* 경로 탐색 또는 AI2-THOR의 ObjectNavExpertAction 사용)
        
        Args:
            object_name: 목표 객체 이름
            target_distance: 목표 거리 (미터, 기본값: 0.5m)
            max_steps: 최대 이동 시도 횟수 (기본값: 50)
            use_astar: A* 알고리즘 사용 여부 (기본값: True)
            
        Returns:
            성공 여부
        """
        object_id = self._find_object_id(object_name)
        if not object_id:
            print(f"✗ Object '{object_name}' not found")
            return False
        
        obj_center = self._get_object_center(object_id)
        if not obj_center or obj_center == {'x': 0.0, 'y': 0.0, 'z': 0.0}:
            print(f"✗ Object '{object_name}' has invalid position")
            return False
        
        dest_pos = [obj_center['x'], obj_center['y'], obj_center['z']]
        
        # A* 경로 탐색 사용
        if use_astar and self.navmesh_graph:
            return self._goto_object_with_astar(object_name, dest_pos, target_distance, max_steps)
        
        # 기존 방식: 가장 가까운 도달 가능한 위치 찾기
        closest_pos = min(
            self.reachable_positions,
            key=lambda p: distance_pts(dest_pos, p)
        )
        return self._goto_object_with_objectnav(object_name, closest_pos, dest_pos, target_distance, max_steps)
    
    def _goto_object_with_astar(self, object_name: str, dest_pos: List[float], target_distance: float, max_steps: int) -> bool:
        """A* 알고리즘을 사용한 경로 탐색 및 이동"""
        # 현재 agent 위치
        metadata = self.controller.last_event.events[self.agent_id].metadata
        robot_pos = metadata["agent"]["position"]
        start_pos = [robot_pos['x'], robot_pos['y'], robot_pos['z']]
        
        # A* 경로 탐색
        print(f"  Finding shortest path using A* algorithm...")
        path = self._astar_pathfinding(start_pos, dest_pos, self.navmesh_graph)
        
        if not path:
            print(f"  ⚠ A* path not found, falling back to ObjectNavExpertAction")
            # 폴백: 가장 가까운 도달 가능한 위치 사용
            closest_pos = min(
                self.reachable_positions,
                key=lambda p: distance_pts(dest_pos, p)
            )
            return self._goto_object_with_objectnav("", closest_pos, dest_pos, target_distance, max_steps)
        
        print(f"  ✓ A* path found: {len(path)} waypoints")
        
        # 경로를 따라 이동
        for waypoint_idx, waypoint in enumerate(path[1:], 1):  # 첫 번째는 현재 위치이므로 스킵
            # ObjectNavExpertAction으로 waypoint까지 이동
            event = self.controller.step(dict(
                action='ObjectNavExpertAction',
                position=dict(x=waypoint[0], y=waypoint[1], z=waypoint[2]),
                agentId=self.agent_id
            ))
            self._capture_frame()
            
            next_action = event.metadata.get('actionReturn')
            if next_action:
                self.controller.step(
                    action=next_action,
                    agentId=self.agent_id,
                    forceAction=True
                )
                self._capture_frame()
            
            time.sleep(0.1)
        
        # A* 경로의 마지막 노드에 도달한 후, 목표까지의 거리 확인
        current_metadata = self.controller.last_event.events[self.agent_id].metadata
        current_pos = current_metadata["agent"]["position"]
        current_distance = distance_pts([current_pos['x'], current_pos['y'], current_pos['z']], dest_pos)
        
        if current_distance <= target_distance:
            print(f"  ✓ Reached destination (distance: {current_distance:.2f}m)")
            return True
        
        # 목표까지 거리가 멀면 추가 이동 시도
        print(f"  → A* path completed, but still {current_distance:.2f}m away. Moving closer to target...")
        
        # 목표에 가장 가까운 도달 가능한 위치 찾기
        closest_reachable = min(
            self.reachable_positions,
            key=lambda p: distance_pts(dest_pos, list(p))
        )
        closest_distance = distance_pts(dest_pos, list(closest_reachable))
        
        # 가장 가까운 도달 가능한 위치까지 추가 이동
        if closest_distance < current_distance:
            # ObjectNavExpertAction으로 가장 가까운 위치까지 이동
            step_count = 0
            while step_count < 20:  # 최대 20스텝
                event = self.controller.step(dict(
                    action='ObjectNavExpertAction',
                    position=dict(x=closest_reachable[0], y=closest_reachable[1], z=closest_reachable[2]),
                    agentId=self.agent_id
                ))
                self._capture_frame()
                
                next_action = event.metadata.get('actionReturn')
                if next_action:
                    self.controller.step(
                        action=next_action,
                        agentId=self.agent_id,
                        forceAction=True
                    )
                    self._capture_frame()
                else:
                    break  # 더 이상 이동할 수 없음
                
                # 현재 거리 확인
                current_metadata = self.controller.last_event.events[self.agent_id].metadata
                current_pos = current_metadata["agent"]["position"]
                current_distance = distance_pts([current_pos['x'], current_pos['y'], current_pos['z']], dest_pos)
                
                if current_distance <= target_distance:
                    print(f"  ✓ Reached destination after additional movement (distance: {current_distance:.2f}m)")
                    return True
                
                step_count += 1
                time.sleep(0.1)
        
        # 최종 확인
        current_metadata = self.controller.last_event.events[self.agent_id].metadata
        current_pos = current_metadata["agent"]["position"]
        final_distance = distance_pts([current_pos['x'], current_pos['y'], current_pos['z']], dest_pos)
        
        if final_distance <= target_distance:
            print(f"  ✓ Reached destination (distance: {final_distance:.2f}m)")
            return True
        else:
            print(f"  ⚠ Reached path end but still {final_distance:.2f}m away from target")
            # 폴백: 기존 ObjectNavExpertAction 방식으로 한 번 더 시도
            return self._goto_object_with_objectnav(object_name, closest_reachable, dest_pos, target_distance, max_steps=20)
    
    def _goto_object_with_objectnav(self, object_name: str, closest_pos: Tuple[float, float, float], dest_pos: List[float], target_distance: float, max_steps: int) -> bool:
        """기존 ObjectNavExpertAction을 사용한 이동 (fallback)"""
        
        step_count = 0
        stuck_count = 0
        prev_distance = float('inf')
        last_successful_action = None
        
        obj_label = f"'{object_name}'" if object_name else "target"
        print(f"  Moving towards {obj_label} (target: {target_distance}m)...")
        
        while step_count < max_steps:
            # 현재 agent 위치
            metadata = self.controller.last_event.events[self.agent_id].metadata
            robot_pos = metadata["agent"]["position"]
            robot_rot = metadata["agent"]["rotation"]["y"]
            
            current_pos = [robot_pos['x'], robot_pos['y'], robot_pos['z']]
            current_distance = distance_pts(current_pos, dest_pos)
            
            # 목표 거리 이내로 도달했는지 확인
            if current_distance <= target_distance:
                print(f"  ✓ Reached '{object_name}' (distance: {current_distance:.2f}m)")
                # 객체를 향해 최종 회전
                self._rotate_towards_object(dest_pos, robot_pos, robot_rot)
                return True
            
            # 이동하지 않는 경우 체크
            distance_change = abs(current_distance - prev_distance)
            if distance_change < 0.05:
                stuck_count += 1
            else:
                stuck_count = 0
            
            prev_distance = current_distance
            
            # 너무 오래 막혀있으면 다른 경로 시도
            if stuck_count > 10:
                # 다른 도달 가능한 위치로 이동 시도
                reachable_sorted = sorted(
                    self.reachable_positions,
                    key=lambda p: distance_pts(dest_pos, p)
                )
                # 두 번째로 가까운 위치 시도
                if len(reachable_sorted) > 1:
                    closest_pos = reachable_sorted[1]
                stuck_count = 0
                print(f"  ⚠ Stuck, trying alternative path...")
            
            # ObjectNavExpertAction 사용 (AI2-THOR의 내장 경로 탐색)
            # 목표 위치를 객체 중심이 아닌 가장 가까운 도달 가능한 위치로 설정
            event = self.controller.step(dict(
                action='ObjectNavExpertAction',
                position=dict(x=closest_pos[0], y=closest_pos[1], z=closest_pos[2]),
                agentId=self.agent_id
            ))
            self._capture_frame()  # 모든 step 후 프레임 캡처
            
            # ObjectNavExpertAction이 반환한 다음 액션 실행
            next_action = event.metadata.get('actionReturn')
            if next_action:
                action_success = self.controller.step(
                    action=next_action, 
                    agentId=self.agent_id, 
                    forceAction=True
                )
                self._capture_frame()  # 모든 step 후 프레임 캡처
                last_successful_action = next_action
                
                # 액션이 성공했는지 확인
                if action_success.metadata.get('lastActionSuccess', False):
                    stuck_count = 0  # 성공하면 stuck 카운터 리셋
            else:
                # ObjectNavExpertAction이 다음 액션을 반환하지 않으면
                # 직접 MoveAhead 시도 (이미 목표에 가까운 경우)
                if current_distance > target_distance:
                    # 객체 방향으로 회전 후 이동
                    robot_object_vec = [dest_pos[0] - robot_pos['x'], dest_pos[2] - robot_pos['z']]
                    if np.linalg.norm(robot_object_vec) > 0.01:
                        y_axis = [0, 1]
                        unit_y = np.array(y_axis) / np.linalg.norm(y_axis)
                        unit_vector = np.array(robot_object_vec) / np.linalg.norm(robot_object_vec)
                        
                        angle = math.atan2(np.linalg.det([unit_vector, unit_y]), np.dot(unit_vector, unit_y))
                        angle = 360 * angle / (2 * math.pi)
                        angle = (angle + 360) % 360
                        rot_angle = angle - robot_rot
                        
                        # 각도 정규화
                        if rot_angle > 180:
                            rot_angle -= 360
                        elif rot_angle < -180:
                            rot_angle += 360
                        
                        # 큰 각도 차이만 회전 (작은 각도는 무시, 최대 90도씩)
                        if abs(rot_angle) > 15:
                            rotation_amount = min(abs(rot_angle), 90)  # 최대 90도씩 회전
                            if rot_angle > 0:
                                self.controller.step(action="RotateRight", degrees=rotation_amount, agentId=self.agent_id)
                            else:
                                self.controller.step(action="RotateLeft", degrees=rotation_amount, agentId=self.agent_id)
                            self._capture_frame()  # 모든 step 후 프레임 캡처
                            time.sleep(0.2)
                        
                        # 앞으로 이동
                        move_event = self.controller.step(action="MoveAhead", moveMagnitude=0.25, agentId=self.agent_id)
                        self._capture_frame()  # 모든 step 후 프레임 캡처
                        if move_event.metadata.get('lastActionSuccess', False):
                            stuck_count = 0
                        else:
                            # 이동 실패 시 다른 방향 시도
                            stuck_count += 1
                            if stuck_count % 3 == 0:
                                # 좌우로 회전하여 장애물 회피
                                self.controller.step(action="RotateRight", degrees=30, agentId=self.agent_id)
                                self._capture_frame()  # 모든 step 후 프레임 캡처
                                time.sleep(0.2)
            
            step_count += 1
            time.sleep(0.2)  # 액션 간 대기 시간
            
            # 진행 상황 출력 (5스텝마다)
            if step_count % 5 == 0:
                print(f"  Distance to '{object_name}': {current_distance:.2f}m (step {step_count}, stuck: {stuck_count})")
        
        # 최종 거리 확인
        metadata = self.controller.last_event.events[self.agent_id].metadata
        robot_pos = metadata["agent"]["position"]
        final_pos = [robot_pos['x'], robot_pos['y'], robot_pos['z']]
        final_distance = distance_pts(final_pos, dest_pos)
        
        # 최종 회전을 위해 rotation 가져오기
        final_metadata = self.controller.last_event.events[self.agent_id].metadata
        final_robot_rot = final_metadata["agent"]["rotation"]["y"]
        
        if final_distance <= target_distance:
            print(f"  ✓ Reached '{object_name}' (distance: {final_distance:.2f}m)")
            self._rotate_towards_object(dest_pos, robot_pos, final_robot_rot)
            return True
        else:
            print(f"  ⚠ Could not reach '{object_name}' within {target_distance}m (final distance: {final_distance:.2f}m)")
            # 최소한 객체를 향해 회전
            self._rotate_towards_object(dest_pos, robot_pos, final_robot_rot)
            return False
    
    def _is_object_visible(self, object_id: str) -> bool:
        """객체가 현재 시야에 보이는지 확인"""
        if not self.controller:
            return False
        
        for obj in self.controller.last_event.metadata.get("objects", []):
            if obj["objectId"] == object_id:
                return obj.get("visible", False)
        return False
    
    def _ensure_object_visible(self, object_id: str, object_name: str, max_rotations: int = 8) -> bool:
        """
        객체가 시야에 정확히 보이도록 회전 (객체를 정확히 향하도록, 효율적인 회전)
        
        Args:
            object_id: 객체 ID
            object_name: 객체 이름 (로그용)
            max_rotations: 최대 회전 횟수 (기본값: 8)
            
        Returns:
            객체가 보이는지 여부
        """
        if not self.controller:
            return False
        
        # 객체 위치 가져오기
        obj_center = self._get_object_center(object_id)
        if not obj_center:
            return False
        
        dest_pos = [obj_center['x'], obj_center['y'], obj_center['z']]
        
        print(f"  Rotating to face '{object_name}'...")
        
        # 먼저 객체가 이미 보이는지 확인
        if self._is_object_visible(object_id):
            # 이미 보이면 정확히 중앙에 오도록 한 번만 미세 조정
            metadata = self.controller.last_event.events[self.agent_id].metadata
            robot_pos = metadata["agent"]["position"]
            robot_rot = metadata["agent"]["rotation"]["y"]
            
            robot_object_vec = [dest_pos[0] - robot_pos['x'], dest_pos[2] - robot_pos['z']]
            if np.linalg.norm(robot_object_vec) > 0.01:
                y_axis = [0, 1]
                unit_y = np.array(y_axis) / np.linalg.norm(y_axis)
                unit_vector = np.array(robot_object_vec) / np.linalg.norm(robot_object_vec)
                
                angle = math.atan2(np.linalg.det([unit_vector, unit_y]), np.dot(unit_vector, unit_y))
                angle = 360 * angle / (2 * math.pi)
                angle = (angle + 360) % 360
                rot_angle = angle - robot_rot
                
                # 각도 정규화
                if rot_angle > 180:
                    rot_angle -= 360
                elif rot_angle < -180:
                    rot_angle += 360
                
                # 5도 이상 차이면 미세 조정
                if abs(rot_angle) > 5:
                    if rot_angle > 0:
                        self.controller.step(action="RotateRight", degrees=min(abs(rot_angle), 20), agentId=self.agent_id)
                    else:
                        self.controller.step(action="RotateLeft", degrees=min(abs(rot_angle), 20), agentId=self.agent_id)
                    self._capture_frame()
                    time.sleep(0.2)
            
            print(f"  ✓ '{object_name}' is already visible")
            return True
        
        # 객체가 보이지 않으면 정확한 각도로 한 번에 회전
        metadata = self.controller.last_event.events[self.agent_id].metadata
        robot_pos = metadata["agent"]["position"]
        robot_rot = metadata["agent"]["rotation"]["y"]
        
        robot_object_vec = [dest_pos[0] - robot_pos['x'], dest_pos[2] - robot_pos['z']]
        if np.linalg.norm(robot_object_vec) < 0.01:
            # 이미 목표 위치에 매우 가까움
            return self._is_object_visible(object_id)
        
        # 객체를 향한 정확한 각도 계산
        y_axis = [0, 1]
        unit_y = np.array(y_axis) / np.linalg.norm(y_axis)
        unit_vector = np.array(robot_object_vec) / np.linalg.norm(robot_object_vec)
        
        angle = math.atan2(np.linalg.det([unit_vector, unit_y]), np.dot(unit_vector, unit_y))
        angle = 360 * angle / (2 * math.pi)
        angle = (angle + 360) % 360
        rot_angle = angle - robot_rot
        
        # 각도 정규화
        if rot_angle > 180:
            rot_angle -= 360
        elif rot_angle < -180:
            rot_angle += 360
        
        # 정확한 각도로 한 번에 회전 (최대 90도씩)
        if abs(rot_angle) > 5:
            # 큰 각도는 여러 번에 나눠서 회전 (최대 90도씩)
            remaining_angle = abs(rot_angle)
            rotation_direction = "RotateRight" if rot_angle > 0 else "RotateLeft"
            
            while remaining_angle > 5 and max_rotations > 0:
                rotation_amount = min(remaining_angle, 90)  # 최대 90도씩
                self.controller.step(action=rotation_direction, degrees=rotation_amount, agentId=self.agent_id)
                self._capture_frame()
                time.sleep(0.2)
                
                # 회전 후 객체가 보이는지 확인
                if self._is_object_visible(object_id):
                    print(f"  ✓ '{object_name}' is now visible")
                    # 미세 조정
                    metadata = self.controller.last_event.events[self.agent_id].metadata
                    robot_rot = metadata["agent"]["rotation"]["y"]
                    rot_angle = angle - robot_rot
                    if rot_angle > 180:
                        rot_angle -= 360
                    elif rot_angle < -180:
                        rot_angle += 360
                    
                    if abs(rot_angle) > 3:
                        if rot_angle > 0:
                            self.controller.step(action="RotateRight", degrees=min(abs(rot_angle), 10), agentId=self.agent_id)
                        else:
                            self.controller.step(action="RotateLeft", degrees=min(abs(rot_angle), 10), agentId=self.agent_id)
                        self._capture_frame()
                        time.sleep(0.15)
                    return True
                
                remaining_angle -= rotation_amount
                max_rotations -= 1
        else:
            # 이미 정확히 향하고 있음
            if self._is_object_visible(object_id):
                print(f"  ✓ '{object_name}' is already visible")
                return True
        
        # 최종 확인
        is_visible = self._is_object_visible(object_id)
        if is_visible:
            print(f"  ✓ '{object_name}' is now visible")
        else:
            print(f"  ⚠ Could not make '{object_name}' visible after rotations")
        
        return is_visible
    
    def _rotate_towards_object(self, dest_pos: List[float], robot_pos: Dict[str, float], robot_rot: float):
        """객체를 향해 회전"""
        robot_object_vec = [dest_pos[0] - robot_pos['x'], dest_pos[2] - robot_pos['z']]
        if np.linalg.norm(robot_object_vec) < 0.01:
            return
        
        y_axis = [0, 1]
        unit_y = np.array(y_axis) / np.linalg.norm(y_axis)
        unit_vector = np.array(robot_object_vec) / np.linalg.norm(robot_object_vec)
        
        angle = math.atan2(np.linalg.det([unit_vector, unit_y]), np.dot(unit_vector, unit_y))
        angle = 360 * angle / (2 * math.pi)
        angle = (angle + 360) % 360
        rot_angle = angle - robot_rot
        
        # 각도 정규화
        if rot_angle > 180:
            rot_angle -= 360
        elif rot_angle < -180:
            rot_angle += 360
        
        if abs(rot_angle) > 5:
            if rot_angle > 0:
                self.controller.step(action="RotateRight", degrees=min(abs(rot_angle), 30), agentId=self.agent_id)
            else:
                self.controller.step(action="RotateLeft", degrees=min(abs(rot_angle), 30), agentId=self.agent_id)
            self._capture_frame()  # 회전 후 프레임 캡처
            time.sleep(0.1)
    
    def pickup_object(self, object_name: str) -> bool:
        """객체 집기"""
        object_id = self._find_object_id(object_name)
        if not object_id:
            print(f"✗ Object '{object_name}' not found")
            return False
        
        # 객체로 이동
        self.goto_object(object_name)
        time.sleep(1)
        self._capture_frame()  # 프레임 캡처
        
        # 객체가 시야에 보이도록 회전
        self._ensure_object_visible(object_id, object_name)
        time.sleep(0.2)
        self._capture_frame()  # 프레임 캡처
        
        # 집기
        event = self.controller.step(
            action="PickupObject",
            objectId=object_id,
            agentId=self.agent_id,
            forceAction=True
        )
        self._capture_frame()  # 액션 실행 후 프레임 캡처
        time.sleep(0.1)  # 프레임 안정화
        
        if event.metadata.get('errorMessage'):
            print(f"✗ Pickup failed: {event.metadata['errorMessage']}")
            return False
        
        self.total_exec += 1
        self.success_exec += 1
        print(f"✓ Picked up: {object_name}")
        return True
    
    def put_object(self, object_name: str, receptacle_name: str) -> bool:
        """객체를 수용체에 놓기"""
        receptacle_id = self._find_object_id(receptacle_name)
        if not receptacle_id:
            print(f"✗ Receptacle '{receptacle_name}' not found")
            return False
        
        # 수용체로 이동
        self.goto_object(receptacle_name)
        time.sleep(1)
        self._capture_frame()  # 프레임 캡처
        
        # 수용체가 시야에 정확히 보이도록 회전 (정확히 수용체를 향하도록)
        self._ensure_object_visible(receptacle_id, receptacle_name)
        time.sleep(0.3)  # 회전 후 안정화 시간
        self._capture_frame()  # 프레임 캡처
        
        # 놓기
        event = self.controller.step(
            action="PutObject",
            objectId=receptacle_id,
            agentId=self.agent_id,
            forceAction=True
        )
        self._capture_frame()  # 액션 실행 후 프레임 캡처
        time.sleep(0.1)  # 프레임 안정화
        
        if event.metadata.get('errorMessage'):
            print(f"✗ Put failed: {event.metadata['errorMessage']}")
            return False
        
        self.total_exec += 1
        self.success_exec += 1
        print(f"✓ Put {object_name} in {receptacle_name}")
        return True
    
    def open_object(self, object_name: str) -> bool:
        """객체 열기"""
        object_id = self._find_object_id(object_name)
        if not object_id:
            print(f"✗ Object '{object_name}' not found")
            return False
        
        # 객체로 이동
        self.goto_object(object_name)
        time.sleep(1)
        self._capture_frame()  # 프레임 캡처
        
        # 객체가 시야에 정확히 보이도록 회전 (정확히 객체를 향하도록)
        self._ensure_object_visible(object_id, object_name)
        time.sleep(0.3)  # 회전 후 안정화 시간
        self._capture_frame()  # 프레임 캡처
        
        # 열기
        event = self.controller.step(
            action="OpenObject",
            objectId=object_id,
            agentId=self.agent_id,
            forceAction=True
        )
        self._capture_frame()  # 액션 실행 후 프레임 캡처
        time.sleep(0.1)  # 프레임 안정화
        
        if event.metadata.get('errorMessage'):
            print(f"✗ Open failed: {event.metadata['errorMessage']}")
            return False
        
        self.total_exec += 1
        self.success_exec += 1
        print(f"✓ Opened: {object_name}")
        return True
    
    def close_object(self, object_name: str) -> bool:
        """객체 닫기"""
        object_id = self._find_object_id(object_name)
        if not object_id:
            print(f"✗ Object '{object_name}' not found")
            return False
        
        # 객체로 이동
        self.goto_object(object_name)
        time.sleep(1)
        self._capture_frame()  # 프레임 캡처
        
        # 객체가 시야에 정확히 보이도록 회전 (정확히 객체를 향하도록)
        self._ensure_object_visible(object_id, object_name)
        time.sleep(0.3)  # 회전 후 안정화 시간
        self._capture_frame()  # 프레임 캡처
        
        # 닫기
        event = self.controller.step(
            action="CloseObject",
            objectId=object_id,
            agentId=self.agent_id,
            forceAction=True
        )
        self._capture_frame()  # 액션 실행 후 프레임 캡처
        time.sleep(0.1)  # 프레임 안정화
        
        if event.metadata.get('errorMessage'):
            print(f"✗ Close failed: {event.metadata['errorMessage']}")
            return False
        
        self.total_exec += 1
        self.success_exec += 1
        print(f"✓ Closed: {object_name}")
        return True
    
    def toggle_on(self, object_name: str) -> bool:
        """객체 켜기"""
        object_id = self._find_object_id(object_name)
        if not object_id:
            print(f"✗ Object '{object_name}' not found")
            return False
        
        # 객체로 이동
        self.goto_object(object_name)
        time.sleep(1)
        self._capture_frame()  # 프레임 캡처
        
        # 객체가 시야에 정확히 보이도록 회전 (정확히 객체를 향하도록)
        self._ensure_object_visible(object_id, object_name)
        time.sleep(0.3)  # 회전 후 안정화 시간
        self._capture_frame()  # 프레임 캡처
        
        # 켜기
        event = self.controller.step(
            action="ToggleObjectOn",
            objectId=object_id,
            agentId=self.agent_id,
            forceAction=True
        )
        self._capture_frame()  # 액션 실행 후 프레임 캡처
        time.sleep(0.1)  # 프레임 안정화
        
        if event.metadata.get('errorMessage'):
            print(f"✗ ToggleOn failed: {event.metadata['errorMessage']}")
            return False
        
        self.total_exec += 1
        self.success_exec += 1
        print(f"✓ Turned on: {object_name}")
        return True
    
    def toggle_off(self, object_name: str) -> bool:
        """객체 끄기"""
        object_id = self._find_object_id(object_name)
        if not object_id:
            print(f"✗ Object '{object_name}' not found")
            return False
        
        # 객체로 이동
        self.goto_object(object_name)
        time.sleep(1)
        self._capture_frame()  # 프레임 캡처
        
        # 객체가 시야에 정확히 보이도록 회전 (정확히 객체를 향하도록)
        self._ensure_object_visible(object_id, object_name)
        time.sleep(0.3)  # 회전 후 안정화 시간
        self._capture_frame()  # 프레임 캡처
        
        # 끄기
        event = self.controller.step(
            action="ToggleObjectOff",
            objectId=object_id,
            agentId=self.agent_id,
            forceAction=True
        )
        self._capture_frame()  # 액션 실행 후 프레임 캡처
        time.sleep(0.1)  # 프레임 안정화
        
        if event.metadata.get('errorMessage'):
            print(f"✗ ToggleOff failed: {event.metadata['errorMessage']}")
            return False
        
        self.total_exec += 1
        self.success_exec += 1
        print(f"✓ Turned off: {object_name}")
        return True
    
    def slice_object(self, object_name: str) -> bool:
        """객체 자르기"""
        object_id = self._find_object_id(object_name)
        if not object_id:
            print(f"✗ Object '{object_name}' not found")
            return False
        
        # 객체로 이동
        self.goto_object(object_name)
        time.sleep(1)
        self._capture_frame()  # 프레임 캡처
        
        # 객체가 시야에 정확히 보이도록 회전 (정확히 객체를 향하도록)
        self._ensure_object_visible(object_id, object_name)
        time.sleep(0.3)  # 회전 후 안정화 시간
        self._capture_frame()  # 프레임 캡처
        
        # 자르기
        event = self.controller.step(
            action="SliceObject",
            objectId=object_id,
            agentId=self.agent_id,
            forceAction=True
        )
        self._capture_frame()  # 액션 실행 후 프레임 캡처
        time.sleep(0.1)  # 프레임 안정화
        
        if event.metadata.get('errorMessage'):
            print(f"✗ Slice failed: {event.metadata['errorMessage']}")
            return False
        
        self.total_exec += 1
        self.success_exec += 1
        print(f"✓ Sliced: {object_name}")
        return True
    
    def clean_object(self, object_name: str) -> bool:
        """객체 청소하기"""
        object_id = self._find_object_id(object_name)
        if not object_id:
            print(f"✗ Object '{object_name}' not found")
            return False
        
        # 객체로 이동
        self.goto_object(object_name)
        time.sleep(1)
        
        # 객체가 시야에 정확히 보이도록 회전 (정확히 객체를 향하도록)
        self._ensure_object_visible(object_id, object_name)
        time.sleep(0.3)  # 회전 후 안정화 시간
        
        # 청소
        event = self.controller.step(
            action="CleanObject",
            objectId=object_id,
            agentId=self.agent_id,
            forceAction=True
        )
        
        if event.metadata.get('errorMessage'):
            print(f"✗ Clean failed: {event.metadata['errorMessage']}")
            return False
        
        self.total_exec += 1
        self.success_exec += 1
        print(f"✓ Cleaned: {object_name}")
        return True
    
    def break_object(self, object_name: str) -> bool:
        """객체 깨기"""
        object_id = self._find_object_id(object_name)
        if not object_id:
            print(f"✗ Object '{object_name}' not found")
            return False
        
        # 객체로 이동
        self.goto_object(object_name)
        time.sleep(1)
        self._capture_frame()  # 프레임 캡처
        
        # 객체가 시야에 정확히 보이도록 회전 (정확히 객체를 향하도록)
        self._ensure_object_visible(object_id, object_name)
        time.sleep(0.3)  # 회전 후 안정화 시간
        self._capture_frame()  # 프레임 캡처
        
        # 깨기
        event = self.controller.step(
            action="BreakObject",
            objectId=object_id,
            agentId=self.agent_id,
            forceAction=True
        )
        self._capture_frame()  # 액션 실행 후 프레임 캡처
        time.sleep(0.1)  # 프레임 안정화
        
        if event.metadata.get('errorMessage'):
            print(f"✗ Break failed: {event.metadata['errorMessage']}")
            return False
        
        self.total_exec += 1
        self.success_exec += 1
        print(f"✓ Broke: {object_name}")
        return True
    
    def drop_hand(self) -> bool:
        """손에 든 객체 놓기"""
        event = self.controller.step(
            action="DropHandObject",
            agentId=self.agent_id,
            forceAction=True
        )
        self._capture_frame()  # 액션 실행 후 프레임 캡처
        time.sleep(0.1)  # 프레임 안정화
        
        if event.metadata.get('errorMessage'):
            print(f"✗ DropHand failed: {event.metadata['errorMessage']}")
            return False
        
        print(f"✓ Dropped hand")
        return True
    
    def parse_action_line(self, line: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        코드 라인에서 액션 파싱
        
        Args:
            line: 코드 라인 (예: "GoToObject('Apple')", "PutObject('Apple', 'Fridge')")
            
        Returns:
            (action, target_object, receptacle) 튜플
        """
        line = line.strip()
        
        # assert 문이나 else: 문은 무시
        if line.startswith("assert") or line.startswith("else:") or not line:
            return None, None, None
        
        # 주석 제거
        if "#" in line:
            line = line.split("#")[0].strip()
        
        # 함수 호출 패턴 매칭
        match = re.match(r'(\w+)\(([^)]*)\)', line)
        if not match:
            return None, None, None
        
        action = match.group(1)
        params = match.group(2)
        
        if not params:
            return action, None, None
        
        # 파라미터 파싱
        params = [p.strip().strip("'\"") for p in params.split(",")]
        
        if len(params) == 1:
            return action, params[0], None
        elif len(params) == 2:
            return action, params[0], params[1]
        else:
            return action, None, None
    
    def execute_action(self, action: str, target_object: Optional[str] = None, receptacle: Optional[str] = None) -> bool:
        """
        액션 실행
        
        Args:
            action: 액션 이름 (GoToObject, PickupObject, PutObject, OpenObject, CloseObject, etc.)
            target_object: 타겟 객체
            receptacle: 수용체 객체 (PutObject 액션의 경우)
            
        Returns:
            성공 여부
        """
        # 새로운 액션 이름 형식으로 매핑 (이전 이름도 호환성을 위해 지원)
        action_map = {
            # 새로운 액션 이름
            "GoToObject": self.goto_object,
            "PickupObject": self.pickup_object,
            "PutObject": self.put_object,
            "OpenObject": self.open_object,
            "CloseObject": self.close_object,
            "ToggleObjectOn": self.toggle_on,
            "ToggleObjectOff": self.toggle_off,
            "SliceObject": self.slice_object,
            "CleanObject": self.clean_object,
            "BreakObject": self.break_object,
            "DropHandObject": self.drop_hand,
            # 이전 액션 이름 (하위 호환성)
            "GoTo": self.goto_object,
            "Pickup": self.pickup_object,
            "Put": self.put_object,
            "Open": self.open_object,
            "Close": self.close_object,
            "ToggleOn": self.toggle_on,
            "ToggleOff": self.toggle_off,
            "Slice": self.slice_object,
            "Clean": self.clean_object,
            "Break": self.break_object,
            "DropHand": self.drop_hand,
        }
        
        if action not in action_map:
            print(f"✗ Unknown action: {action}")
            return False
        
        func = action_map[action]
        
        try:
            # PutObject 또는 Put 액션 처리
            if (action == "PutObject" or action == "Put") and receptacle:
                return func(target_object, receptacle)
            # DropHandObject 또는 DropHand 액션 처리
            elif action == "DropHandObject" or action == "DropHand":
                return func()
            elif target_object:
                return func(target_object)
            else:
                print(f"✗ Missing parameters for action: {action}")
                return False
        except Exception as e:
            print(f"✗ Error executing {action}: {str(e)}")
            return False
    
    def _check_assert_condition(self, assert_line: str) -> bool:
        """
        assert 조건 확인
        
        Args:
            assert_line: assert 문 (예: "assert('close' to 'Book')", "assert('Book' in 'hands')")
            
        Returns:
            조건 만족 여부
        """
        assert_line = assert_line.strip()
        if not assert_line.startswith("assert"):
            return False
        
        # assert('...') 형식에서 조건 추출
        match = re.search(r"assert\(['\"](.*?)['\"]\)", assert_line)
        if not match:
            return False
        
        condition = match.group(1).strip()
        
        # 'close' to 'Object' 형식
        if "'close' to" in condition or '"close" to' in condition:
            obj_match = re.search(r"to\s+['\"](.*?)['\"]", condition)
            if obj_match:
                obj_name = obj_match.group(1)
                return self._check_close_to_object(obj_name)
        
        # 'Object' in 'hands' 형식
        if " in 'hands'" in condition or ' in "hands"' in condition:
            obj_match = re.search(r"['\"](.*?)['\"]\s+in", condition)
            if obj_match:
                obj_name = obj_match.group(1)
                return self._check_object_in_hands(obj_name)
        
        # 'Object' is 'opened' 형식
        if " is 'opened'" in condition or ' is "opened"' in condition:
            obj_match = re.search(r"['\"](.*?)['\"]\s+is", condition)
            if obj_match:
                obj_name = obj_match.group(1)
                return self._check_object_opened(obj_name)
        
        # 'Object' is 'closed' 형식
        if " is 'closed'" in condition or ' is "closed"' in condition:
            obj_match = re.search(r"['\"](.*?)['\"]\s+is", condition)
            if obj_match:
                obj_name = obj_match.group(1)
                return self._check_object_closed(obj_name)
        
        # 'Object' visible 형식
        if " visible" in condition:
            obj_match = re.search(r"['\"](.*?)['\"]\s+visible", condition)
            if obj_match:
                obj_name = obj_match.group(1)
                return self._check_object_visible(obj_name)
        
        return False
    
    def _check_close_to_object(self, object_name: str, threshold: float = 1.5) -> bool:
        """객체와 가까운지 확인 (threshold 미터 이내)"""
        if not self.controller:
            return False
        
        object_id = self._find_object_id(object_name)
        if not object_id:
            return False
        
        obj_center = self._get_object_center(object_id)
        if not obj_center:
            return False
        
        metadata = self.controller.last_event.events[self.agent_id].metadata
        agent_pos = metadata["agent"]["position"]
        
        distance = distance_pts(
            [agent_pos['x'], agent_pos['y'], agent_pos['z']],
            [obj_center['x'], obj_center['y'], obj_center['z']]
        )
        
        return distance <= threshold
    
    def _check_object_in_hands(self, object_name: str) -> bool:
        """손에 객체를 들고 있는지 확인"""
        if not self.controller:
            return False
        
        metadata = self.controller.last_event.events[self.agent_id].metadata
        inventory_objects = metadata.get("inventoryObjects", [])
        
        if not inventory_objects:
            return False
        
        # inventory에 있는 객체의 objectId 확인
        for inv_obj in inventory_objects:
            if object_name.lower() in inv_obj.lower():
                return True
        
        return False
    
    def _check_object_opened(self, object_name: str) -> bool:
        """객체가 열려있는지 확인"""
        if not self.controller:
            return False
        
        object_id = self._find_object_id(object_name)
        if not object_id:
            return False
        
        for obj in self.controller.last_event.metadata.get("objects", []):
            if obj["objectId"] == object_id:
                return obj.get("isOpen", False)
        
        return False
    
    def _check_object_closed(self, object_name: str) -> bool:
        """객체가 닫혀있는지 확인"""
        if not self.controller:
            return False
        
        object_id = self._find_object_id(object_name)
        if not object_id:
            return False
        
        for obj in self.controller.last_event.metadata.get("objects", []):
            if obj["objectId"] == object_id:
                return not obj.get("isOpen", True)  # isOpen이 False이거나 없으면 닫혀있음
        
        return False
    
    def _check_object_visible(self, object_name: str) -> bool:
        """객체가 보이는지 확인"""
        if not self.controller:
            return False
        
        object_id = self._find_object_id(object_name)
        if not object_id:
            return False
        
        for obj in self.controller.last_event.metadata.get("objects", []):
            if obj["objectId"] == object_id:
                return obj.get("visible", False)
        
        return False
    
    def _parse_else_blocks(self, lines: List[str], start_idx: int) -> Tuple[List[str], int]:
        """
        else: 블록(들)의 액션들 추출 (여러 개의 else: 블록 지원)
        
        Args:
            lines: 전체 코드 라인 리스트
            start_idx: 첫 번째 else: 라인의 인덱스
            
        Returns:
            (else 블록의 액션 라인 리스트, 다음 라인 인덱스) 튜플
        """
        else_actions = []
        i = start_idx
        
        # 첫 번째 else: 라인의 들여쓰기 레벨 확인
        first_else_line = lines[i]
        else_indent = len(first_else_line) - len(first_else_line.lstrip())
        
        while i < len(lines):
            line = lines[i]
            line_stripped = line.strip()
            
            # 빈 라인이거나 주석만 있으면 스킵
            if not line_stripped or line_stripped.startswith("#"):
                i += 1
                continue
            
            # else: 라인인 경우
            if line_stripped.startswith("else:"):
                # else: 다음 라인부터 액션 추출
                j = i + 1
                while j < len(lines):
                    next_line = lines[j]
                    next_line_stripped = next_line.strip()
                    
                    # 빈 라인이거나 주석만 있으면 스킵
                    if not next_line_stripped or next_line_stripped.startswith("#"):
                        j += 1
                        continue
                    
                    # 현재 라인의 들여쓰기
                    next_line_indent = len(next_line) - len(next_line.lstrip())
                    
                    # 들여쓰기가 else 블록보다 작거나 같으면 이 else 블록 종료
                    if next_line_indent <= else_indent:
                        break
                    
                    # 액션 라인인지 확인
                    action, _, _ = self.parse_action_line(next_line)
                    if action:
                        else_actions.append(next_line.strip())
                    
                    j += 1
                
                i = j
            else:
                # else:가 아닌 라인이 나오면 종료
                break
        
        return else_actions, i
    
    def execute_program(self, program_code: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        ProgPrompt 형식의 프로그램 실행 (assert/else 처리 포함)
        
        Args:
            program_code: 프로그램 코드 문자열
            max_retries: assert 실패 시 최대 재시도 횟수
            
        Returns:
            실행 결과 딕셔너리
        """
        if not self.controller:
            self.initialize()
        
        lines = program_code.split("\n")
        executed_actions = []
        failed_actions = []
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # 빈 라인이나 주석은 스킵
            if not line or line.startswith("#"):
                i += 1
                continue
            
            # assert 문 처리
            if line.startswith("assert"):
                print(f"\n[Checking] {line}")
                condition_met = self._check_assert_condition(line)
                
                if condition_met:
                    print(f"  ✓ Condition met, skipping recovery")
                    # assert 다음 라인으로 이동 (else 블록 스킵)
                    i += 1
                    # else: 블록이 있으면 스킵
                    for j in range(i, len(lines)):
                        if lines[j].strip().startswith("else:"):
                            # else 블록의 끝까지 찾기
                            else_indent = len(lines[j]) - len(lines[j].lstrip())
                            k = j + 1
                            while k < len(lines):
                                next_line = lines[k]
                                next_line_stripped = next_line.strip()
                                if next_line_stripped and not next_line_stripped.startswith("#"):
                                    next_line_indent = len(next_line) - len(next_line.lstrip())
                                    if next_line_indent <= else_indent:
                                        break
                                k += 1
                            i = k
                            break
                    continue
                else:
                    print(f"  ✗ Condition not met, executing recovery actions")
                    # 다음 else: 블록 찾기
                    else_idx = None
                    for j in range(i + 1, len(lines)):
                        if lines[j].strip().startswith("else:"):
                            else_idx = j
                            break
                    
                    if else_idx is not None:
                        # else 블록(들)의 액션들 추출
                        else_actions, next_idx = self._parse_else_blocks(lines, else_idx)
                        
                        # else 블록의 액션들 실행
                        for else_action_line in else_actions:
                            action, target_obj, receptacle = self.parse_action_line(else_action_line)
                            if action:
                                print(f"  [Recovery] {else_action_line}")
                                success = self.execute_action(action, target_obj, receptacle)
                                
                                executed_actions.append({
                                    "line": else_action_line,
                                    "action": action,
                                    "target_object": target_obj,
                                    "receptacle": receptacle,
                                    "success": success,
                                    "type": "recovery"
                                })
                                
                                if not success:
                                    failed_actions.append(else_action_line)
                                
                                time.sleep(0.3)
                        
                        # else 블록 다음으로 이동
                        i = next_idx
                        continue
                    else:
                        print(f"  ⚠ No recovery action found")
                        i += 1
                        continue
            
            # else: 문은 assert 처리 중에 이미 처리되므로 스킵
            if line.startswith("else:"):
                i += 1
                continue
            
            # 일반 액션 실행
            action, target_obj, receptacle = self.parse_action_line(lines[i])
            if action:
                print(f"\n[Executing] {line}")
                success = self.execute_action(action, target_obj, receptacle)
                
                executed_actions.append({
                    "line": line,
                    "action": action,
                    "target_object": target_obj,
                    "receptacle": receptacle,
                    "success": success,
                    "type": "main"
                })
                
                if not success:
                    failed_actions.append(line)
                
                time.sleep(0.3)
            
            i += 1
        
        # 프로그램 종료 시 최종 프레임 캡처
        if self.save_video:
            self._capture_frame()
        
        result = {
            "total_actions": len(executed_actions),
            "successful_actions": sum(1 for a in executed_actions if a["success"]),
            "failed_actions": len(failed_actions),
            "executed_actions": executed_actions,
            "failed_actions": failed_actions,
            "success_rate": (sum(1 for a in executed_actions if a["success"]) / len(executed_actions) * 100) if executed_actions else 0,
        }
        
        # 비디오 경로 추가
        if self.save_video and hasattr(self, 'video_path'):
            result["video_path"] = self.video_path
        
        return result
    
    def get_available_objects(self) -> List[Dict[str, Any]]:
        """
        AI2THOR 환경에서 사용 가능한 모든 객체 목록과 속성을 가져오기
        
        Returns:
            객체 목록 (각 객체는 objectType, properties 등을 포함)
        """
        if not self.controller:
            return []
        
        metadata = self.controller.last_event.metadata
        objects = []
        seen_types = set()
        
        for obj in metadata.get("objects", []):
            obj_type = normalize_object_type(obj.get("objectType", ""))
            
            # 중복 제거 (같은 타입의 객체는 한 번만 추가)
            if obj_type and obj_type not in seen_types:
                seen_types.add(obj_type)
                
                # 객체 속성 추출
                properties = []
                if obj.get("pickupable", False):
                    properties.append("Pickupable")
                if obj.get("receptacle", False):
                    properties.append("Receptacle")
                if obj.get("openable", False):
                    properties.append("Openable")
                if obj.get("toggleable", False):
                    properties.append("Toggleable")
                if obj.get("sliceable", False):
                    properties.append("Sliceable")
                if obj.get("breakable", False):
                    properties.append("Breakable")
                if obj.get("fillable", False):
                    properties.append("Fillable")
                if obj.get("dirtyable", False):
                    properties.append("Dirty")
                if obj.get("canFillWithLiquid", False):
                    properties.append("Fillable")
                
                # On/Off 상태 확인
                if obj.get("isToggled", False) or obj.get("isOn", False):
                    properties.append("On")
                else:
                    if obj.get("toggleable", False) or obj.get("isToggled") is not None:
                        properties.append("Off")
                
                objects.append({
                    "objectType": obj_type,
                    "properties": properties,
                    "objectId": obj.get("objectId", ""),
                })
        
        return objects
    
    def get_available_actions(self) -> List[str]:
        """
        AI2THOR 환경에서 사용 가능한 액션 목록 가져오기
        
        Returns:
            액션 목록 (info.txt 형식)
        """
        actions = [
            "GoToObject <obj>              # Navigate close to an object",
            "PickupObject <obj>            # Pick up a pickupable object (agent can hold only ONE object at a time)",
            "PutObject <obj> <recp>        # Place held object inside/on receptacle",
            "OpenObject <obj>              # Open openable container/appliance",
            "CloseObject <obj>             # Close openable object",
            "SliceObject <obj>             # Slice sliceable food while holding a Knife",
            "ToggleObjectOn <obj>          # Turn on toggleable object (Faucet, Microwave, Toaster, Dishwasher, CoffeeMachine, StoveBurner)",
            "ToggleObjectOff <obj>         # Turn off toggleable object",
            "CleanObject <obj>             # Wash/clean an object near Sink/Faucet",
            "BreakObject <obj>             # Break breakable object",
            "DropHandObject                # Drop whatever the agent is holding (required before picking up another object)",
            "ThrowObject <obj>             # Throw an object",
            "PushObject <obj> <obj>        # Push an object",
            "PullObject <obj> <obj>        # Pull an object",
            "UseUpObject <obj>             # Use up an object",
            "FillObjectWithLiquid <obj> <liquid>     # Fill an object with a liquid",
            "EmptyLiquidFromObject <obj>             # Empty an object",
            "DirtyObject <obj>             # Make an object dirty",

        ]
        return actions
    
    def export_to_info_txt_format(self, output_path: Optional[str] = None) -> str:
        """
        AI2THOR 환경의 객체와 액션을 info.txt 형식으로 내보내기
        
        Args:
            output_path: 저장할 파일 경로 (None이면 문자열만 반환)
        
        Returns:
            info.txt 형식의 문자열
        """
        objects = self.get_available_objects()
        actions = self.get_available_actions()
        
        lines = ["ACTION TEMPLATES:", ""]
        
        # 액션 목록 추가 (info.txt 형식으로 변환)
        action_map = {
            "GoToObject": "goto <obj>",
            "PickupObject": "pickup <obj>",
            "PutObject": "put <obj><recp>",
            "OpenObject": "open <obj>",
            "CloseObject": "close <obj>",
            "SliceObject": "slice <obj>",
            "ToggleObjectOn": "toggleon <obj>",
            "ToggleObjectOff": "toggleoff <obj>",
            "CleanObject": "clean <obj>",
            "BreakObject": "break <obj>",
            "DropHandObject": "drop",
            "ThrowObject": "throw <obj>",
            "PushObject": "push <obj> <obj>",
            "PullObject": "pull <obj> <obj>",
            "UseUpObject": "useup <obj>",
            "FillObjectWithLiquid": "fill <obj> <liquid>",
            "EmptyLiquidFromObject": "empty <obj>",
            "DirtyObject": "dirty <obj>",
        }
        
        for action_line in actions:
            action_name = action_line.split()[0]
            if action_name in action_map:
                lines.append(action_map[action_name])
        
        lines.extend(["done", "", "All OBJECTS and PROPERTIES:", ""])
        
        # 객체 목록 추가
        for obj in sorted(objects, key=lambda x: x["objectType"]):
            obj_type = obj["objectType"]
            props = obj["properties"]
            if props:
                props_str = ", ".join(props)
                lines.append(f"{obj_type} : {props_str}")
            else:
                lines.append(obj_type)
        
        result = "\n".join(lines)
        
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result)
            print(f"✓ Exported AI2THOR objects and actions to {output_path}")
        
        return result
    
    def get_scene_state(self) -> Dict[str, Any]:
        """현재 scene 상태 가져오기"""
        if not self.controller:
            return {}
        
        metadata = self.controller.last_event.metadata
        return {
            "agent_position": metadata["agent"]["position"],
            "agent_rotation": metadata["agent"]["rotation"],
            "objects": [
                {
                    "objectId": obj["objectId"],
                    "objectType": normalize_object_type(obj.get("objectType", "")),
                    "position": obj.get("position", {}),
                    "visible": obj.get("visible", False),
                    "isPickedUp": obj.get("isPickedUp", False),
                    "isOpen": obj.get("isOpen", False),
                }
                for obj in metadata.get("objects", [])
            ],
        }
    
    def close(self):
        """리소스 정리"""
        # 비디오 writer 종료
        if self.save_video:
            self._close_video_writer()
        
        if self.controller:
            self.controller.stop()
            print("✓ AI2-THOR closed")


# 사용 예시
if __name__ == "__main__":
    executor = AI2ThorExecutor(scene="FloorPlan1", headless=False)
    executor.initialize()
    
    # ProgPrompt 형식 프로그램 (assert/else 포함)
    program = """
    def break_mug():
        \t# Step 1: Find and pick up the mug
        \tGoToObject('Mug')
        \tassert('close' to 'Mug')
        \t\telse: GoToObject('Mug')
        \tPickupObject('Mug')
        \tBreakObject('Mug')


    """
    
    result = executor.execute_program(program)
    print(f"\nExecution Summary:")
    print(f"  Total Actions: {result['total_actions']}")
    print(f"  Successful: {result['successful_actions']}")
    print(f"  Failed: {result['failed_actions']}")
    print(f"  Success Rate: {result['success_rate']:.1f}%")
    
    print(f"\nExecuted Actions:")
    for action in result['executed_actions']:
        status = "✓" if action['success'] else "✗"
        action_type = action.get('type', 'main')
        print(f"  {status} [{action_type}] {action['line']}")
    
    executor.close()