#!/usr/bin/env python3
"""
Scene Graph에서 특정 object 정보를 찾아서 출력하는 스크립트
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional


def load_scene_graph(scene_number: int) -> Optional[Dict[str, Any]]:
    """
    Scene Graph JSON 파일 로드
    
    Args:
        scene_number: FloorPlan 번호
        
    Returns:
        Scene Graph 딕셔너리 또는 None
    """
    # 가능한 경로들
    possible_paths = [
        f"scripts/scene_graph_structured_FloorPlan{scene_number}.json",
        f"scene_graph_structured_FloorPlan{scene_number}.json",
    ]
    
    for path_str in possible_paths:
        path = Path(path_str)
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    scene_graph = json.load(f)
                print(f"✓ Scene Graph 로드 완료: {path.absolute()}")
                return scene_graph
            except Exception as e:
                print(f"❌ Scene Graph 로드 실패: {e}")
                return None
    
    print(f"❌ Scene Graph 파일을 찾을 수 없습니다:")
    for path_str in possible_paths:
        print(f"   - {Path(path_str).absolute()}")
    return None


def find_objects(scene_graph: Dict[str, Any], object_name: str) -> List[Dict[str, Any]]:
    """
    Scene Graph에서 object 이름과 일치하는 객체들 찾기
    
    Args:
        scene_graph: Scene Graph 딕셔너리
        object_name: 찾을 object 이름 (대소문자 구분 없음)
        
    Returns:
        일치하는 객체들의 리스트
    """
    objects = scene_graph.get("nodes", {}).get("objects", [])
    object_name_lower = object_name.lower()
    
    matched_objects = []
    for obj in objects:
        obj_type = obj.get("objectType", "")
        obj_id = obj.get("nodeId", "")
        
        # objectType 또는 nodeId에 object_name이 포함되어 있는지 확인
        if (object_name_lower in obj_type.lower() or 
            object_name_lower in obj_id.lower()):
            matched_objects.append(obj)
    
    return matched_objects


def print_object_info(obj: Dict[str, Any], index: int = None):
    """
    Object 정보를 보기 좋게 출력
    
    Args:
        obj: Object 딕셔너리
        index: 객체 인덱스 (여러 개일 경우)
    """
    prefix = f"[{index}] " if index is not None else ""
    
    print(f"\n{'='*80}")
    print(f"{prefix}Object 정보")
    print(f"{'='*80}")
    
    # 기본 정보
    print(f"nodeId: {obj.get('nodeId', 'N/A')}")
    print(f"objectType: {obj.get('objectType', 'N/A')}")
    
    # 위치 정보
    position = obj.get('position', {})
    if position:
        print(f"\n📍 위치:")
        print(f"  X: {position.get('x', 0):.3f} m")
        print(f"  Y: {position.get('y', 0):.3f} m")
        print(f"  Z: {position.get('z', 0):.3f} m")
    
    # 회전 정보
    rotation = obj.get('rotation', {})
    if rotation:
        print(f"\n🔄 회전:")
        print(f"  X: {rotation.get('x', 0):.1f}°")
        print(f"  Y: {rotation.get('y', 0):.1f}°")
        print(f"  Z: {rotation.get('z', 0):.1f}°")
    
    # 속성 정보
    print(f"\n📋 속성:")
    attributes = []
    if obj.get('pickupable', False):
        attributes.append("pickupable")
    if obj.get('openable', False):
        attributes.append("openable")
    if obj.get('receptacle', False):
        attributes.append("receptacle")
    if obj.get('toggleable', False):
        attributes.append("toggleable")
    if obj.get('sliceable', False):
        attributes.append("sliceable")
    if obj.get('breakable', False):
        attributes.append("breakable")
    if obj.get('cookable', False):
        attributes.append("cookable")
    if attributes:
        print(f"  {', '.join(attributes)}")
    else:
        print(f"  (속성 없음)")
    
    # 상태 정보
    print(f"\n🔹 상태:")
    states = []
    if obj.get('isOpen', False):
        openness = obj.get('openness', 0.0)
        states.append(f"isOpen=True (openness={openness:.2f})")
    else:
        states.append("isOpen=False")
    
    if obj.get('isPickedUp', False):
        states.append("isPickedUp=True")
    else:
        states.append("isPickedUp=False")
    
    if obj.get('isToggled', False):
        states.append("isToggled=True")
    
    if obj.get('isSliced', False):
        states.append("isSliced=True")
    
    if obj.get('isBroken', False):
        states.append("isBroken=True")
    
    if obj.get('isCooked', False):
        states.append("isCooked=True")
    
    for state in states:
        print(f"  {state}")
    
    # 거리 및 가시성
    distance = obj.get('distance', None)
    visible = obj.get('visible', False)
    print(f"\n👁️  가시성:")
    print(f"  visible: {visible}")
    if distance is not None:
        print(f"  distance: {distance:.3f} m")
    
    # 부모 수용체
    parent_receptacles = obj.get('parentReceptacles', [])
    if parent_receptacles:
        print(f"\n📦 부모 수용체:")
        for recp_id in parent_receptacles:
            print(f"  - {recp_id}")
    else:
        print(f"\n📦 부모 수용체: 없음")
    
    # 기타 정보
    mass = obj.get('mass', None)
    temperature = obj.get('temperature', None)
    if mass is not None:
        print(f"\n⚖️  질량: {mass:.3f} kg")
    if temperature:
        print(f"🌡️  온도: {temperature}")
    
    print(f"{'='*80}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="Scene Graph에서 특정 object 정보를 찾아서 출력"
    )
    parser.add_argument(
        "--scene-number",
        type=int,
        help="FloorPlan 번호 (예: 1, 201, 301, 401)"
    )
    parser.add_argument(
        "--object",
        type=str,
        help="찾을 object 이름 (예: Apple, Fridge, CounterTop)"
    )
    
    args = parser.parse_args()
    
    # Scene 번호 입력 받기
    if args.scene_number is None:
        try:
            scene_number = int(input("FloorPlan 번호를 입력하세요 (예: 1, 201, 301, 401): "))
        except (ValueError, KeyboardInterrupt):
            print("❌ 잘못된 입력입니다. 숫자를 입력해주세요.")
            return
    else:
        scene_number = args.scene_number
    
    # Object 이름 입력 받기
    if args.object is None:
        object_name = input("Object 이름을 입력하세요 (예: Apple, Fridge): ").strip()
        if not object_name:
            print("❌ Object 이름이 입력되지 않았습니다.")
            return
    else:
        object_name = args.object
    
    print(f"\n🔍 Scene {scene_number}에서 '{object_name}' 객체 검색 중...\n")
    
    # Scene Graph 로드
    scene_graph = load_scene_graph(scene_number)
    if scene_graph is None:
        return
    
    # Scene 정보 출력
    metadata = scene_graph.get("metadata", {})
    scene_name = metadata.get("sceneName", "Unknown")
    total_objects = metadata.get("totalObjects", 0)
    print(f"Scene: {scene_name}")
    print(f"Total Objects: {total_objects}")
    print()
    
    # Object 찾기
    matched_objects = find_objects(scene_graph, object_name)
    
    if not matched_objects:
        print(f"❌ '{object_name}'와 일치하는 객체를 찾을 수 없습니다.")
        return
    
    print(f"✓ {len(matched_objects)}개의 객체를 찾았습니다.\n")
    
    # 각 객체 정보 출력
    if len(matched_objects) == 1:
        print_object_info(matched_objects[0])
    else:
        for i, obj in enumerate(matched_objects, 1):
            print_object_info(obj, index=i)
        
        print(f"\n총 {len(matched_objects)}개의 객체가 발견되었습니다.")


if __name__ == "__main__":
    main()
