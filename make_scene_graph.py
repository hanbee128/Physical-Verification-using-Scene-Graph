#!/usr/bin/env python3
"""
Scene Graph 생성 스크립트

scene_graph_FloorPlan1.json 파일을 LLM이 물리적 검증에 활용할 수 있는 
구조화된 Scene Graph 형식으로 변환합니다.

Scene Graph 구조:
- Nodes: Agent Node, Object Nodes
- Edges: HOLDS, IN, VISIBLE, REACHABLE
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Any, Optional

# 상수 정의
REACH_THRESHOLD = 1.39  # REACHABLE 판정 기준 거리 (미터) - PICKUP_DISTANCE_THRESHOLD
HAND_SPHERE_RADIUS = 0.5  # Hand sphere 반경 (미터)


def distance(pos1: Dict[str, float], pos2: Dict[str, float]) -> float:
    """두 위치 사이의 3D 유클리드 거리 계산"""
    dx = pos1.get("x", 0) - pos2.get("x", 0)
    dy = pos1.get("y", 0) - pos2.get("y", 0)
    dz = pos1.get("z", 0) - pos2.get("z", 0)
    return math.sqrt(dx**2 + dy**2 + dz**2)


def create_agent_node(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Agent Node 생성
    
    AgentNode {
        position: {x, y, z},
        rotation: {x, y, z},
        cameraHorizon: float,
        isHolding: bool,
        heldObjectId: Optional[str]
    }
    """
    agent = metadata.get("agent", {})
    inventory_objects = metadata.get("inventoryObjects", [])
    held_object_pose = metadata.get("heldObjectPose", {})
    
    # heldObjectId 확인 (inventoryObjects 또는 isPickedUp 상태에서)
    held_object_id = None
    if inventory_objects:
        if isinstance(inventory_objects, list) and len(inventory_objects) > 0:
            held_object_id = inventory_objects[0]
        elif isinstance(inventory_objects, str):
            held_object_id = inventory_objects
    
    # isPickedUp 상태에서도 확인 (더 정확함)
    if not held_object_id:
        for obj in metadata.get("objects", []):
            if obj.get("isPickedUp", False):
                held_object_id = obj.get("objectId")
                break
    
    # heldObjectPose가 있으면 holding 상태로 간주
    if not held_object_id and held_object_pose:
        # heldObjectPose가 있으면 무언가를 들고 있을 가능성이 높음
        # 하지만 정확한 objectId는 objects에서 찾아야 함
        for obj in metadata.get("objects", []):
            if obj.get("isPickedUp", False):
                held_object_id = obj.get("objectId")
                break
    
    agent_node = {
        "nodeType": "Agent",
        "nodeId": "agent_0",
        "position": agent.get("position", {"x": 0, "y": 0, "z": 0}),
        "rotation": agent.get("rotation", {"x": 0, "y": 0, "z": 0}),
        "cameraHorizon": agent.get("cameraHorizon", 0.0),
        "isHolding": held_object_id is not None,
        "heldObjectId": held_object_id,
        "heldObjectPose": held_object_pose if held_object_pose else None
    }
    
    return agent_node


def create_object_node(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Object Node 생성
    
    ObjectNode {
        objectId,
        objectType,
        pickupable,
        openable,
        receptacle,
        toggleable,
        isOpen,
        isToggled,
        visible,
        distance,
        position,
        parentReceptacles,
        controlledObjects,
        ...
    }
    """
    object_node = {
        "nodeType": "Object",
        "nodeId": obj.get("objectId", ""),
        "objectType": obj.get("objectType", ""),
        "pickupable": obj.get("pickupable", False),
        "openable": obj.get("openable", False),
        "receptacle": obj.get("receptacle", False),
        "toggleable": obj.get("toggleable", False),
        "isOpen": obj.get("isOpen", False),
        "isToggled": obj.get("isToggled", False),
        "visible": obj.get("visible", False),
        "distance": obj.get("distance", float('inf')),
        "position": obj.get("position", {"x": 0, "y": 0, "z": 0}),
        "rotation": obj.get("rotation", {"x": 0, "y": 0, "z": 0}),
        "parentReceptacles": obj.get("parentReceptacles", []),
        "controlledObjects": obj.get("controlledObjects", []),
        "isPickedUp": obj.get("isPickedUp", False),
        "receptacleObjectIds": obj.get("receptacleObjectIds", []),
        "mass": obj.get("mass", 0.0),
        "sliceable": obj.get("sliceable", False),
        "isSliced": obj.get("isSliced", False),
        "breakable": obj.get("breakable", False),
        "isBroken": obj.get("isBroken", False),
        "cookable": obj.get("cookable", False),
        "isCooked": obj.get("isCooked", False),
        "temperature": obj.get("temperature", "RoomTemp"),
        "openness": obj.get("openness", 0.0)
    }
    
    return object_node


def create_edges(
    agent_node: Dict[str, Any],
    object_nodes: List[Dict[str, Any]],
    reach_threshold: float = REACH_THRESHOLD
) -> List[Dict[str, Any]]:
    """
    Edge 생성 (Scene Graph의 핵심)
    
    생성되는 Edge 타입:
    1. HOLDS(agent, obj) - obj.isPickedUp == True일 때
    2. IN(obj, receptacle) - parentReceptacles 기반
    3. VISIBLE(agent, obj) - obj.visible == True일 때
    4. REACHABLE(agent, obj) - obj.visible and obj.distance < reach_threshold일 때
    """
    edges = []
    agent_id = agent_node.get("nodeId", "agent_0")
    agent_pos = agent_node.get("position", {})
    
    # 객체 ID로 빠른 조회를 위한 딕셔너리 생성
    object_dict = {obj.get("nodeId"): obj for obj in object_nodes}
    
    for obj_node in object_nodes:
        obj_id = obj_node.get("nodeId", "")
        obj_type = obj_node.get("objectType", "")
        
        # Edge 1: HOLDS(agent, obj)
        # if obj.isPickedUp == True: HOLDS(agent, obj)
        if obj_node.get("isPickedUp", False):
            edges.append({
                "edgeType": "HOLDS",
                "source": agent_id,
                "target": obj_id,
                "sourceType": "Agent",
                "targetType": "Object",
                "targetObjectType": obj_type
            })
        
        # Edge 2: IN(obj, receptacle)
        # for r in obj.parentReceptacles: IN(obj, r)
        parent_receptacles = obj_node.get("parentReceptacles", [])
        if parent_receptacles:
            for receptacle_id in parent_receptacles:
                # receptacle이 object_nodes에 존재하는지 확인
                if receptacle_id in object_dict:
                    receptacle_node = object_dict[receptacle_id]
                    edges.append({
                        "edgeType": "IN",
                        "source": obj_id,
                        "target": receptacle_id,
                        "sourceType": "Object",
                        "targetType": "Object",
                        "sourceObjectType": obj_type,
                        "targetObjectType": receptacle_node.get("objectType", "")
                    })
        
        # Edge 3: VISIBLE(agent, obj)
        # if obj.visible == True: VISIBLE(agent, obj)
        if obj_node.get("visible", False):
            edges.append({
                "edgeType": "VISIBLE",
                "source": agent_id,
                "target": obj_id,
                "sourceType": "Agent",
                "targetType": "Object",
                "targetObjectType": obj_type,
                "distance": obj_node.get("distance", float('inf'))
            })
        
        # Edge 4: REACHABLE(agent, obj)
        # if obj.visible and obj.distance < reach_threshold: REACHABLE(agent, obj)
        obj_visible = obj_node.get("visible", False)
        obj_distance = obj_node.get("distance", float('inf'))
        
        if obj_visible and obj_distance < reach_threshold:
            # 추가 검증: 실제 3D 거리 계산
            obj_pos = obj_node.get("position", {})
            if obj_pos and agent_pos:
                actual_distance = distance(agent_pos, obj_pos)
                if actual_distance < reach_threshold:
                    edges.append({
                        "edgeType": "REACHABLE",
                        "source": agent_id,
                        "target": obj_id,
                        "sourceType": "Agent",
                        "targetType": "Object",
                        "targetObjectType": obj_type,
                        "distance": obj_distance,
                        "actualDistance": actual_distance,
                        "reachThreshold": reach_threshold
                    })
    
    return edges


def create_scene_graph(input_file: str, output_file: str, reach_threshold: float = REACH_THRESHOLD) -> Dict[str, Any]:
    """
    Scene Graph 생성 메인 함수
    
    Args:
        input_file: 입력 JSON 파일 경로 (scene_graph_FloorPlan1.json)
        output_file: 출력 Scene Graph JSON 파일 경로
        reach_threshold: REACHABLE 판정 기준 거리 (미터)
        
    Returns:
        생성된 Scene Graph 딕셔너리
    """
    print(f"📖 입력 파일 읽는 중: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # 4.1 Node 생성
    print("🔹 Agent Node 생성 중...")
    agent_node = create_agent_node(metadata)
    
    print(f"🔹 Object Nodes 생성 중... (총 {len(metadata.get('objects', []))}개 객체)")
    object_nodes = []
    for obj in metadata.get("objects", []):
        object_node = create_object_node(obj)
        object_nodes.append(object_node)
    
    print(f"   ✓ {len(object_nodes)}개의 Object Node 생성 완료")
    
    # 4.2 Edge 생성
    print("🔹 Edges 생성 중...")
    edges = create_edges(agent_node, object_nodes, reach_threshold)
    
    # Edge 타입별 통계
    edge_stats = {}
    for edge in edges:
        edge_type = edge.get("edgeType", "UNKNOWN")
        edge_stats[edge_type] = edge_stats.get(edge_type, 0) + 1
    
    print(f"   ✓ {len(edges)}개의 Edge 생성 완료")
    print(f"   Edge 통계:")
    for edge_type, count in edge_stats.items():
        print(f"     - {edge_type}: {count}개")
    
    # Scene Graph 구조화
    scene_graph = {
        "metadata": {
            "sceneName": metadata.get("sceneName", "Unknown"),
            "agentId": metadata.get("agentId", 0),
            "reachThreshold": reach_threshold,
            "handSphereRadius": HAND_SPHERE_RADIUS,
            "totalObjects": len(object_nodes),
            "totalEdges": len(edges)
        },
        "nodes": {
            "agent": agent_node,
            "objects": object_nodes
        },
        "edges": edges,
        "edgeStatistics": edge_stats
    }
    
    # 출력 파일 저장
    print(f"💾 Scene Graph 저장 중: {output_file}")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(scene_graph, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Scene Graph 생성 완료!")
    print(f"   - Agent Node: 1개")
    print(f"   - Object Nodes: {len(object_nodes)}개")
    print(f"   - Edges: {len(edges)}개")
    print(f"   - 저장 위치: {output_path.absolute()}")
    
    return scene_graph


def main():
    """메인 함수"""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description="Scene Graph 생성 스크립트: FloorPlan 번호를 입력받아 scene graph를 생성"
    )
    
    parser.add_argument(
        "floorplan_number",
        type=int,
        nargs="?",
        help="FloorPlan 번호 (예: 1, 201, 301, 401)"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="입력 JSON 파일 경로 (지정하지 않으면 floorplan_number 기반으로 자동 생성)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="출력 Scene Graph JSON 파일 경로 (지정하지 않으면 floorplan_number 기반으로 자동 생성)"
    )
    
    parser.add_argument(
        "--reach-threshold",
        type=float,
        default=REACH_THRESHOLD,
        help=f"REACHABLE 판정 기준 거리 (미터, 기본값: {REACH_THRESHOLD})"
    )
    
    args = parser.parse_args()
    
    # FloorPlan 번호 입력 받기
    if args.floorplan_number is None:
        try:
            floorplan_number = int(input("FloorPlan 번호를 입력하세요 (예: 1, 201, 301, 401): "))
        except (ValueError, KeyboardInterrupt):
            print("❌ 잘못된 입력입니다. 숫자를 입력해주세요.")
            sys.exit(1)
    else:
        floorplan_number = args.floorplan_number
    
    # 입력 파일 경로 결정
    if args.input:
        input_file = args.input
    else:
        # 루트 디렉토리와 scripts 디렉토리 모두 확인
        possible_paths = [
            f"scene_graph_FloorPlan{floorplan_number}.json",
            f"scripts/scene_graph_FloorPlan{floorplan_number}.json"
        ]
        input_file = None
        for path in possible_paths:
            if Path(path).exists():
                input_file = path
                break
        
        if input_file is None:
            print(f"❌ 오류: scene_graph_FloorPlan{floorplan_number}.json 파일을 찾을 수 없습니다.")
            print(f"   다음 경로를 확인했습니다:")
            for path in possible_paths:
                print(f"     - {Path(path).absolute()}")
            sys.exit(1)
    
    # 출력 파일 경로 결정
    if args.output:
        output_file = args.output
    else:
        output_file = f"scripts/scene_graph_structured_FloorPlan{floorplan_number}.json"
    
    print(f"🔍 FloorPlan {floorplan_number} 처리 중...")
    print(f"   입력 파일: {input_file}")
    print(f"   출력 파일: {output_file}")
    print()
    
    # Scene Graph 생성
    scene_graph = create_scene_graph(
        input_file=input_file,
        output_file=output_file,
        reach_threshold=args.reach_threshold
    )
    
    # 간단한 통계 출력
    print("\n" + "=" * 80)
    print("Scene Graph 통계")
    print("=" * 80)
    print(f"Scene: {scene_graph['metadata']['sceneName']}")
    print(f"Agent Node: 1개")
    print(f"Object Nodes: {scene_graph['metadata']['totalObjects']}개")
    print(f"Total Edges: {scene_graph['metadata']['totalEdges']}개")
    print("\nEdge 타입별 통계:")
    for edge_type, count in scene_graph['edgeStatistics'].items():
        print(f"  {edge_type}: {count}개")
    print("=" * 80)


if __name__ == "__main__":
    main()
