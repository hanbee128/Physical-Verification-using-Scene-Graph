#!/usr/bin/env python3
"""
Scene Graph에서 에이전트 현재 정보와 목표 객체의 정보를 출력하는 스크립트

사용법:
    python scene_graph_extractor.py --object "Apple"
    python scene_graph_extractor.py --object "Apple" --scene-graph scripts/scene_graph_structured.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional


def load_scene_graph(scene_graph_path: str) -> Dict[str, Any]:
    """Scene Graph JSON 파일 로드"""
    try:
        with open(scene_graph_path, "r", encoding="utf-8") as f:
            scene_graph = json.load(f)
        return scene_graph
    except FileNotFoundError:
        print(f"❌ Scene Graph 파일을 찾을 수 없습니다: {scene_graph_path}")
        return {}
    except json.JSONDecodeError as e:
        print(f"❌ JSON 파싱 오류: {e}")
        return {}
    except Exception as e:
        print(f"❌ Scene Graph 로드 실패: {e}")
        return {}


def find_target_object(scene_graph: Dict[str, Any], object_name: str) -> List[Dict[str, Any]]:
    """
    목표 객체 이름으로 Scene Graph에서 객체 찾기 (정확한 매칭 우선, 부분 매칭 fallback)
    
    Args:
        scene_graph: Scene Graph 딕셔너리
        object_name: 찾을 객체 이름 (예: "Apple", "Fridge", "Knife")
        
    Returns:
        매칭된 객체 노드 리스트 (정확한 매칭이 우선)
    """
    object_nodes = scene_graph.get("nodes", {}).get("objects", [])
    exact_matches = []
    partial_matches = []
    
    object_name_lower = object_name.lower()
    
    # 제외할 객체 타입 리스트 (예: 'Knife'를 찾을 때 'ButterKnife'는 제외)
    exclude_types = []
    if object_name_lower == "knife":
        exclude_types = ["butterknife"]
    
    for obj_node in object_nodes:
        obj_type = obj_node.get("objectType", "")
        obj_id = obj_node.get("nodeId", "")
        obj_type_lower = obj_type.lower()
        obj_id_lower = obj_id.lower()
        
        # 제외할 타입인지 확인 (부분 문자열 포함 여부 확인)
        # "knife"를 찾을 때 "butterknife"는 무조건 제외
        if object_name_lower == "knife":
            if "butter" in obj_type_lower or "butter" in obj_id_lower:
                continue
        
        # 정확한 매칭 우선 (타입이 정확히 일치)
        if obj_type_lower == object_name_lower:
            exact_matches.append(obj_node)
        # nodeId에서도 정확한 매칭 확인 (nodeId가 "Knife|..." 형식인 경우)
        elif obj_id_lower.startswith(object_name_lower + "|") or obj_id_lower == object_name_lower:
            exact_matches.append(obj_node)
        # 정확한 매칭이 없으면 부분 매칭 (단, 제외 타입은 이미 필터링됨)
        elif object_name_lower in obj_type_lower or object_name_lower in obj_id_lower:
            # 추가 확인: "knife"를 찾을 때 "butterknife"는 제외 (이중 체크)
            if object_name_lower == "knife" and ("butter" in obj_type_lower or "butter" in obj_id_lower):
                continue
            partial_matches.append(obj_node)
    
    # "Knife"를 찾을 때는 정확한 매칭만 반환 (부분 매칭 반환 안 함)
    if object_name_lower == "knife":
        return exact_matches
    
    # 정확한 매칭이 있으면 그것만 반환, 없으면 부분 매칭 반환
    return exact_matches if exact_matches else partial_matches


def get_related_edges(scene_graph: Dict[str, Any], object_node: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    특정 객체와 관련된 엣지 찾기
    
    Args:
        scene_graph: Scene Graph 딕셔너리
        object_node: 대상 객체 노드
        
    Returns:
        관련 엣지 리스트
    """
    edges = scene_graph.get("edges", [])
    object_id = object_node.get("nodeId", "")
    related_edges = []
    
    for edge in edges:
        source = edge.get("source", "")
        target = edge.get("target", "")
        
        # 객체가 source 또는 target인 엣지 찾기
        if object_id in [source, target]:
            related_edges.append(edge)
    
    return related_edges


def print_agent_info(agent_node: Dict[str, Any]):
    """에이전트 정보 출력"""
    print("=" * 80)
    print("🤖 에이전트 정보")
    print("=" * 80)
    
    node_id = agent_node.get("nodeId", "N/A")
    position = agent_node.get("position", {})
    rotation = agent_node.get("rotation", {})
    camera_horizon = agent_node.get("cameraHorizon", 0.0)
    is_holding = agent_node.get("isHolding", False)
    held_object_id = agent_node.get("heldObjectId", None)
    held_object_pose = agent_node.get("heldObjectPose", {})
    
    print(f"Node ID: {node_id}")
    print(f"Position: ({position.get('x', 0):.3f}, {position.get('y', 0):.3f}, {position.get('z', 0):.3f})")
    print(f"Rotation: ({rotation.get('x', 0):.1f}°, {rotation.get('y', 0):.1f}°, {rotation.get('z', 0):.1f}°)")
    print(f"Camera Horizon: {camera_horizon:.1f}°")
    print(f"Is Holding: {is_holding}")
    
    if is_holding and held_object_id:
        print(f"Held Object ID: {held_object_id}")
        if held_object_pose:
            held_pos = held_object_pose.get("position", {})
            held_rot = held_object_pose.get("rotation", {})
            print(f"  Held Object Position: ({held_pos.get('x', 0):.3f}, {held_pos.get('y', 0):.3f}, {held_pos.get('z', 0):.3f})")
            print(f"  Held Object Rotation: ({held_rot.get('x', 0):.1f}°, {held_rot.get('y', 0):.1f}°, {held_rot.get('z', 0):.1f}°)")
    else:
        print("Held Object: None")
    
    print()


def print_object_info(object_node: Dict[str, Any], related_edges: List[Dict[str, Any]], scene_graph: Optional[Dict[str, Any]] = None):
    """목표 객체 정보 출력"""
    print("=" * 80)
    print(f"📦 객체 정보: {object_node.get('objectType', 'N/A')}")
    print("=" * 80)
    
    # 기본 정보
    node_id = object_node.get("nodeId", "N/A")
    object_type = object_node.get("objectType", "N/A")
    position = object_node.get("position", {})
    rotation = object_node.get("rotation", {})
    
    print(f"Node ID: {node_id}")
    print(f"Object Type: {object_type}")
    print(f"Position: ({position.get('x', 0):.3f}, {position.get('y', 0):.3f}, {position.get('z', 0):.3f})")
    print(f"Rotation: ({rotation.get('x', 0):.3f}, {rotation.get('y', 0):.3f}, {rotation.get('z', 0):.3f})")
    print()
    
    # 속성 정보 (True인 것만 출력, distance는 항상 출력)
    print("속성:")
    if object_node.get('pickupable', False):
        print(f"  pickupable: True")
    if object_node.get('openable', False):
        print(f"  openable: True")
    if object_node.get('receptacle', False):
        print(f"  receptacle: True")
    if object_node.get('toggleable', False):
        print(f"  toggleable: True")
    if object_node.get('visible', False):
        print(f"  visible: True")
    # distance는 항상 출력
    print(f"  distance: {object_node.get('distance', 0):.3f}m")
    print()
    
    # 상태 정보
    print("상태:")
    print(f"  isOpen: {object_node.get('isOpen', False)}")
    print(f"  isToggled: {object_node.get('isToggled', False)}")
    print(f"  isPickedUp: {object_node.get('isPickedUp', False)}")
    if object_node.get('openable', False):
        print(f"  openness: {object_node.get('openness', 0.0):.2f}")
    print()
    
    # 부모 수용체 정보
    parent_receptacles = object_node.get("parentReceptacles", [])
    if parent_receptacles:
        print(f"Parent Receptacles ({len(parent_receptacles)}):")
        for recp_id in parent_receptacles:
            print(f"  - {recp_id}")
            
            # 부모 수용체의 속성 정보 출력
            if scene_graph:
                # Scene Graph에서 부모 수용체 노드 찾기
                recp_node = None
                object_nodes = scene_graph.get("nodes", {}).get("objects", [])
                for obj_node in object_nodes:
                    if obj_node.get("nodeId") == recp_id:
                        recp_node = obj_node
                        break
                
                if recp_node:
                    recp_type = recp_node.get("objectType", "N/A")
                    recp_pos = recp_node.get("position", {})
                    
                    print(f"    타입: {recp_type}")
                    print(f"    위치: ({recp_pos.get('x', 0):.3f}, {recp_pos.get('y', 0):.3f}, {recp_pos.get('z', 0):.3f})")
                    
                    # 속성 정보 (True인 것만)
                    recp_attributes = []
                    if recp_node.get('pickupable', False):
                        recp_attributes.append("pickupable")
                    if recp_node.get('openable', False):
                        recp_attributes.append("openable")
                    if recp_node.get('receptacle', False):
                        recp_attributes.append("receptacle")
                    if recp_node.get('toggleable', False):
                        recp_attributes.append("toggleable")
                    if recp_attributes:
                        print(f"    속성: {', '.join(recp_attributes)}")
                    
                    # 상태 정보
                    recp_states = []
                    if recp_node.get('isOpen', False):
                        recp_states.append("isOpen")
                    if recp_node.get('isToggled', False):
                        recp_states.append("isToggled")
                    if recp_node.get('isPickedUp', False):
                        recp_states.append("isPickedUp")
                    if recp_states:
                        print(f"    상태: {', '.join(recp_states)}")
                    if recp_node.get('openable', False):
                        print(f"    openness: {recp_node.get('openness', 0.0):.2f}")
                else:
                    print(f"    (Scene Graph에서 노드를 찾을 수 없음)")
        print()
    
    # 수용 가능한 객체 정보
    receptacle_object_ids = object_node.get("receptacleObjectIds", [])
    if receptacle_object_ids:
        print(f"Receptacle Object IDs ({len(receptacle_object_ids)}):")
        for obj_id in receptacle_object_ids:
            print(f"  - {obj_id}")
        print()
    
    # 관련 엣지 정보
    if related_edges:
        print(f"관련 엣지 ({len(related_edges)}):")
        for edge in related_edges:
            edge_type = edge.get("edgeType", "UNKNOWN")
            source = edge.get("source", "N/A")
            target = edge.get("target", "N/A")
            
            print(f"  - {edge_type}: {source} → {target}")
            
            # 엣지 타입별 추가 정보
            if edge_type == "VISIBLE":
                distance = edge.get("distance", 0)
                print(f"    거리: {distance:.3f}m")
            elif edge_type == "REACHABLE":
                distance = edge.get("distance", 0)
                reach_threshold = edge.get("reachThreshold", 1.39)
                print(f"    거리: {distance:.3f}m (임계값: {reach_threshold}m)")
            elif edge_type == "HOLDS":
                print(f"    Agent가 이 객체를 들고 있음")
            elif edge_type == "IN":
                print(f"    객체가 수용체 안에 있음")
            elif edge_type == "ON":
                print(f"    객체가 수용체 위에 있음")
        print()
    else:
        print("관련 엣지: 없음")
        print()


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="Scene Graph에서 에이전트와 목표 객체 정보를 출력하는 스크립트"
    )
    
    parser.add_argument(
        "--object",
        type=str,
        required=True,
        help="목표 객체 이름 (예: 'Apple', 'Fridge')"
    )
    
    parser.add_argument(
        "--scene-graph",
        type=str,
        default="scripts/scene_graph_structured.json",
        help="Scene Graph JSON 파일 경로"
    )
    
    args = parser.parse_args()
    
    # Scene Graph 파일 경로 처리
    scene_graph_path = Path(args.scene_graph)
    if not scene_graph_path.is_absolute():
        # 상대 경로인 경우 스크립트 디렉토리 기준으로 변환
        script_dir = Path(__file__).parent.parent
        scene_graph_path = script_dir / scene_graph_path
    
    # Scene Graph 로드
    print(f"📂 Scene Graph 로드 중: {scene_graph_path}")
    scene_graph = load_scene_graph(str(scene_graph_path))
    
    if not scene_graph:
        print("❌ Scene Graph를 로드할 수 없습니다.")
        return
    
    print("✓ Scene Graph 로드 완료\n")
    
    # 에이전트 정보 출력
    agent_node = scene_graph.get("nodes", {}).get("agent", {})
    if agent_node:
        print_agent_info(agent_node)
    else:
        print("⚠️  에이전트 노드를 찾을 수 없습니다.\n")
    
    # 목표 객체 찾기
    print(f"🔍 목표 객체 검색 중: '{args.object}'")
    matched_objects = find_target_object(scene_graph, args.object)
    
    if not matched_objects:
        print(f"❌ '{args.object}'와 일치하는 객체를 찾을 수 없습니다.")
        print("\n사용 가능한 객체 타입:")
        object_nodes = scene_graph.get("nodes", {}).get("objects", [])
        object_types = sorted(set([obj.get("objectType", "Unknown") for obj in object_nodes]))
        for obj_type in object_types[:20]:  # 처음 20개만 출력
            print(f"  - {obj_type}")
        if len(object_types) > 20:
            print(f"  ... 외 {len(object_types) - 20}개")
        return
    
    print(f"✓ {len(matched_objects)}개의 객체를 찾았습니다.\n")
    
    # 각 매칭된 객체 정보 출력
    for i, obj_node in enumerate(matched_objects, 1):
        if len(matched_objects) > 1:
            print(f"\n{'=' * 80}")
            print(f"객체 {i}/{len(matched_objects)}")
            print(f"{'=' * 80}\n")
        
        # 관련 엣지 찾기
        related_edges = get_related_edges(scene_graph, obj_node)
        
        # 객체 정보 출력
        print_object_info(obj_node, related_edges, scene_graph)
        
        if i < len(matched_objects):
            print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
