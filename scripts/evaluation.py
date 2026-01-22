#!/usr/bin/env python3
"""
완전 자동화된 평가 지표 수집 스크립트

평가 지표:
1. Exec (Executability): 실행 중 lastActionSuccess=False가 한 번이라도 있으면 Exec=0
2. SR (Success Rate): 최종 상태 metadata에서 목표 조건 검사
3. TCR (Task Completion Rate): 각 task별 goal condition 체크
4. GCR (Goal Condition Rate): 각 goal condition이 만족된 비율
5. Plan Length: 최종 실행된 액션 수
6. Guard Trigger Distribution: 어떤 guard가 실패했는지 누적 count
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
from collections import defaultdict


def check_goal_condition(
    action_type: str,
    action_args: Dict[str, Any],
    final_metadata: Dict[str, Any],
    scene_graph: Optional[Dict[str, Any]] = None
) -> bool:
    """
    최종 상태에서 목표 조건 검사 (SR 계산용)
    
    Args:
        action_type: 액션 타입 (예: "PutObject", "OpenObject")
        action_args: 액션 인자 (예: {"o": "Apple", "r": "Fridge"})
        final_metadata: 최종 AI2-THOR metadata
        scene_graph: 최종 Scene Graph (선택사항)
        
    Returns:
        목표 조건 만족 여부
    """
    if scene_graph:
        # Scene Graph 기반 검사 (더 정확)
        return check_goal_condition_with_scene_graph(action_type, action_args, scene_graph)
    else:
        # Metadata 기반 검사 (fallback)
        return check_goal_condition_with_metadata(action_type, action_args, final_metadata)


def check_goal_condition_with_scene_graph(
    action_type: str,
    action_args: Dict[str, Any],
    scene_graph: Dict[str, Any]
) -> bool:
    """Scene Graph 기반 목표 조건 검사"""
    object_name = action_args.get("o")
    receptacle_name = action_args.get("r")
    
    if action_type == "PutObject":
        # Put X in Y → Y in X.parentReceptacles
        if not object_name or not receptacle_name:
            return False
        
        # Scene Graph에서 객체 찾기
        object_nodes = scene_graph.get("nodes", {}).get("objects", [])
        target_obj = None
        for obj in object_nodes:
            obj_id = obj.get("nodeId", "")
            obj_type = obj.get("objectType", "")
            if object_name.lower() in obj_id.lower() or object_name.lower() in obj_type.lower():
                target_obj = obj
                break
        
        if not target_obj:
            return False
        
        # parentReceptacles 확인
        parent_receptacles = target_obj.get("parentReceptacles", [])
        if not parent_receptacles:
            return False
        
        # receptacle_name이 parentReceptacles에 있는지 확인
        for recp_id in parent_receptacles:
            if receptacle_name.lower() in recp_id.lower():
                return True
        
        return False
    
    elif action_type == "OpenObject":
        # Open Y → Y.isOpen == True
        if not object_name:
            return False
        
        object_nodes = scene_graph.get("nodes", {}).get("objects", [])
        for obj in object_nodes:
            obj_id = obj.get("nodeId", "")
            obj_type = obj.get("objectType", "")
            if object_name.lower() in obj_id.lower() or object_name.lower() in obj_type.lower():
                return obj.get("isOpen", False)
        
        return False
    
    elif action_type == "CloseObject":
        # Close Y → Y.isOpen == False
        if not object_name:
            return False
        
        object_nodes = scene_graph.get("nodes", {}).get("objects", [])
        for obj in object_nodes:
            obj_id = obj.get("nodeId", "")
            obj_type = obj.get("objectType", "")
            if object_name.lower() in obj_id.lower() or object_name.lower() in obj_type.lower():
                return not obj.get("isOpen", False)
        
        return False
    
    elif action_type == "PickupObject":
        # Pickup X → agent.isHolding == X
        if not object_name:
            return False
        
        agent_node = scene_graph.get("nodes", {}).get("agent", {})
        if not agent_node.get("isHolding", False):
            return False
        
        held_object_id = agent_node.get("heldObjectId", "")
        if not held_object_id:
            return False
        
        # held_object_id에서 object_name 확인
        return object_name.lower() in held_object_id.lower()
    
    elif action_type == "ToggleObjectOn":
        # ToggleOn Y → Y.isToggled == True
        if not object_name:
            return False
        
        object_nodes = scene_graph.get("nodes", {}).get("objects", [])
        for obj in object_nodes:
            obj_id = obj.get("nodeId", "")
            obj_type = obj.get("objectType", "")
            if object_name.lower() in obj_id.lower() or object_name.lower() in obj_type.lower():
                return obj.get("isToggled", False)
        
        return False
    
    elif action_type == "ToggleObjectOff":
        # ToggleOff Y → Y.isToggled == False
        if not object_name:
            return False
        
        object_nodes = scene_graph.get("nodes", {}).get("objects", [])
        for obj in object_nodes:
            obj_id = obj.get("nodeId", "")
            obj_type = obj.get("objectType", "")
            if object_name.lower() in obj_id.lower() or object_name.lower() in obj_type.lower():
                return not obj.get("isToggled", False)
        
        return False
    
    elif action_type == "SliceObject":
        # Slice X → X.isSliced == True
        if not object_name:
            return False
        
        object_nodes = scene_graph.get("nodes", {}).get("objects", [])
        for obj in object_nodes:
            obj_id = obj.get("nodeId", "")
            obj_type = obj.get("objectType", "")
            if object_name.lower() in obj_id.lower() or object_name.lower() in obj_type.lower():
                return obj.get("isSliced", False)
        
        return False
    
    elif action_type == "BreakObject":
        # Break X → X.isBroken == True
        if not object_name:
            return False
        
        object_nodes = scene_graph.get("nodes", {}).get("objects", [])
        for obj in object_nodes:
            obj_id = obj.get("nodeId", "")
            obj_type = obj.get("objectType", "")
            if object_name.lower() in obj_id.lower() or object_name.lower() in obj_type.lower():
                return obj.get("isBroken", False)
        
        return False
    
    # GoToObject는 목표 조건이 없음 (이동만 수행)
    return True


def check_goal_condition_with_metadata(
    action_type: str,
    action_args: Dict[str, Any],
    metadata: Dict[str, Any]
) -> bool:
    """Metadata 기반 목표 조건 검사 (fallback)"""
    # Metadata 기반 검사는 Scene Graph보다 덜 정확하므로
    # 가능하면 Scene Graph를 사용하는 것이 좋음
    objects = metadata.get("objects", [])
    agent = metadata.get("agent", {})
    
    object_name = action_args.get("o")
    receptacle_name = action_args.get("r")
    
    if action_type == "PutObject":
        # Put X in Y → Y in X.parentReceptacles
        if not object_name or not receptacle_name:
            return False
        
        for obj in objects:
            obj_id = obj.get("objectId", "")
            if object_name.lower() in obj_id.lower():
                parent_receptacles = obj.get("parentReceptacles", [])
                for recp_id in parent_receptacles:
                    if receptacle_name.lower() in recp_id.lower():
                        return True
        
        return False
    
    elif action_type == "OpenObject":
        # Open Y → Y.isOpen == True
        if not object_name:
            return False
        
        for obj in objects:
            obj_id = obj.get("objectId", "")
            if object_name.lower() in obj_id.lower():
                return obj.get("isOpen", False)
        
        return False
    
    elif action_type == "CloseObject":
        # Close Y → Y.isOpen == False
        if not object_name:
            return False
        
        for obj in objects:
            obj_id = obj.get("objectId", "")
            if object_name.lower() in obj_id.lower():
                return not obj.get("isOpen", False)
        
        return False
    
    elif action_type == "PickupObject":
        # Pickup X → agent.isHolding == X
        if not object_name:
            return False
        
        if not agent.get("isHolding", False):
            return False
        
        held_object_id = agent.get("heldObjectId", "")
        if not held_object_id:
            return False
        
        return object_name.lower() in held_object_id.lower()
    
    # 기타 액션들은 metadata에서 확인 어려움
    return True


def parse_tasks_from_instruction(instruction: str) -> List[Dict[str, Any]]:
    """
    자연어 instruction에서 task를 원자 단위로 파싱
    
    예:
    Instruction: Put Apple, Tomato and Potato into SinkBasin
    → Task set = [
        {"type": "PutObject", "args": {"o": "Apple", "r": "SinkBasin"}},
        {"type": "PutObject", "args": {"o": "Tomato", "r": "SinkBasin"}},
        {"type": "PutObject", "args": {"o": "Potato", "r": "SinkBasin"}}
    ]
    """
    tasks = []
    
    # 간단한 패턴 매칭 (더 정교한 파싱은 필요시 개선)
    # Put X, Y, Z into R
    put_match = re.search(r'put\s+([^,]+(?:,\s*[^,]+)*)\s+into\s+(\w+)', instruction.lower())
    if put_match:
        objects_str = put_match.group(1)
        receptacle = put_match.group(2)
        objects = [obj.strip() for obj in objects_str.split(',')]
        for obj in objects:
            tasks.append({
                "type": "PutObject",
                "args": {"o": obj, "r": receptacle}
            })
    
    # Open X
    open_match = re.search(r'open\s+(\w+)', instruction.lower())
    if open_match:
        obj = open_match.group(1)
        tasks.append({
            "type": "OpenObject",
            "args": {"o": obj}
        })
    
    # Close X
    close_match = re.search(r'close\s+(\w+)', instruction.lower())
    if close_match:
        obj = close_match.group(1)
        tasks.append({
            "type": "CloseObject",
            "args": {"o": obj}
        })
    
    # Pickup X
    pickup_match = re.search(r'pickup\s+(\w+)', instruction.lower())
    if pickup_match:
        obj = pickup_match.group(1)
        tasks.append({
            "type": "PickupObject",
            "args": {"o": obj}
        })
    
    # Slice X
    slice_match = re.search(r'slice\s+(\w+)', instruction.lower())
    if slice_match:
        obj = slice_match.group(1)
        tasks.append({
            "type": "SliceObject",
            "args": {"o": obj}
        })
    
    # ToggleOn X
    toggle_on_match = re.search(r'toggle\s+on\s+(\w+)', instruction.lower())
    if toggle_on_match:
        obj = toggle_on_match.group(1)
        tasks.append({
            "type": "ToggleObjectOn",
            "args": {"o": obj}
        })
    
    # ToggleOff X
    toggle_off_match = re.search(r'toggle\s+off\s+(\w+)', instruction.lower())
    if toggle_off_match:
        obj = toggle_off_match.group(1)
        tasks.append({
            "type": "ToggleObjectOff",
            "args": {"o": obj}
        })
    
    return tasks


def evaluate_episode(
    episode_id: str,
    instruction: str,
    executed_actions: List[Dict[str, Any]],
    final_metadata: Optional[Dict[str, Any]] = None,
    final_scene_graph: Optional[Dict[str, Any]] = None,
    failed_guards: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    단일 에피소드 평가 지표 계산
    
    Args:
        episode_id: 에피소드 ID
        instruction: 자연어 instruction
        executed_actions: 실행된 액션 리스트 [{"line": str, "action": str, "target_object": str, "receptacle": str, "success": bool, "lastActionSuccess": bool, "errorMessage": str}, ...]
        final_metadata: 최종 AI2-THOR metadata
        final_scene_graph: 최종 Scene Graph
        failed_guards: 실패한 guard 리스트 (검증 단계에서 수집)
        
    Returns:
        평가 지표 딕셔너리
    """
    # 1. Exec (Executability): 실행 중 lastActionSuccess=False가 한 번이라도 있으면 Exec=0
    exec_value = 1
    for action in executed_actions:
        # lastActionSuccess가 명시적으로 False인 경우
        if action.get("lastActionSuccess") is False:
            exec_value = 0
            break
        # success가 False이고 lastActionSuccess가 없는 경우 (추정)
        if not action.get("success", True) and action.get("lastActionSuccess") is None:
            exec_value = 0
            break
    
    # 2. Plan Length: 최종 실행된 액션 수
    plan_length = len(executed_actions)
    
    # 3. Guard Trigger Distribution: 어떤 guard가 실패했는지 누적 count
    guard_distribution = defaultdict(int)
    if failed_guards:
        for guard in failed_guards:
            guard_distribution[guard] += 1
    
    # 4. SR (Success Rate): 최종 상태에서 목표 조건 검사
    # 실행된 액션 중 목표가 있는 액션들에 대해 검사
    successful_actions = 0
    total_goal_actions = 0
    
    for action in executed_actions:
        action_type = action.get("action")
        if not action_type:
            continue
        
        # 목표 조건이 있는 액션만 검사
        if action_type in ["PutObject", "OpenObject", "CloseObject", "PickupObject", 
                          "ToggleObjectOn", "ToggleObjectOff", "SliceObject", "BreakObject"]:
            total_goal_actions += 1
            action_args = {
                "o": action.get("target_object"),
                "r": action.get("receptacle")
            }
            
            if check_goal_condition(action_type, action_args, final_metadata or {}, final_scene_graph):
                successful_actions += 1
    
    sr = (successful_actions / total_goal_actions * 100) if total_goal_actions > 0 else 0.0
    
    # 5. TCR (Task Completion Rate): 각 task별 goal condition 체크
    tasks = parse_tasks_from_instruction(instruction)
    completed_tasks = 0
    
    for task in tasks:
        if check_goal_condition(task["type"], task["args"], final_metadata or {}, final_scene_graph):
            completed_tasks += 1
    
    tcr = (completed_tasks / len(tasks) * 100) if len(tasks) > 0 else 0.0
    
    # 6. GCR (Goal Condition Rate): 각 goal condition이 만족된 비율
    # 실행된 액션에서 goal condition을 추출하고 각각 검사
    goal_conditions = []
    satisfied_conditions = 0
    
    for action in executed_actions:
        action_type = action.get("action")
        if not action_type:
            continue
        
        # 목표 조건이 있는 액션만 검사
        if action_type in ["PutObject", "OpenObject", "CloseObject", "PickupObject", 
                          "ToggleObjectOn", "ToggleObjectOff", "SliceObject", "BreakObject"]:
            action_args = {
                "o": action.get("target_object"),
                "r": action.get("receptacle")
            }
            
            # Goal condition 생성 (액션 타입과 인자로 고유하게 식별)
            goal_condition = {
                "type": action_type,
                "args": action_args,
                "action_line": action.get("line", "")
            }
            goal_conditions.append(goal_condition)
            
            # Goal condition 만족 여부 확인
            if check_goal_condition(action_type, action_args, final_metadata or {}, final_scene_graph):
                satisfied_conditions += 1
    
    gcr = (satisfied_conditions / len(goal_conditions) * 100) if len(goal_conditions) > 0 else 0.0
    
    # 결과 반환
    return {
        "episode_id": episode_id,
        "instruction": instruction,
        "Exec": exec_value,
        "SR": sr,
        "TCR": tcr,
        "GCR": gcr,
        "PlanLength": plan_length,
        "failed_guards": dict(guard_distribution),
        "missing_tasks": [task for task in tasks if not check_goal_condition(
            task["type"], task["args"], final_metadata or {}, final_scene_graph
        )],
        "total_goal_actions": total_goal_actions,
        "successful_goal_actions": successful_actions,
        "total_tasks": len(tasks),
        "completed_tasks": completed_tasks,
        "total_goal_conditions": len(goal_conditions),
        "satisfied_goal_conditions": satisfied_conditions
    }


def load_execution_result(execution_result_file: Path) -> Dict[str, Any]:
    """
    실행 결과 파일 로드
    
    실행 결과 파일 형식:
    {
        "episode_id": "...",
        "instruction": "...",
        "executed_actions": [...],
        "final_metadata": {...},
        "final_scene_graph": {...},
        "failed_guards": [...]
    }
    """
    with open(execution_result_file, "r", encoding="utf-8") as f:
        return json.load(f)


def aggregate_episode_results(episode_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    여러 에피소드 결과를 집계
    
    Args:
        episode_results: 에피소드 결과 리스트
        
    Returns:
        집계 결과
    """
    total_episodes = len(episode_results)
    if total_episodes == 0:
        return {}
    
    # 평균 계산
    avg_exec = sum(r["Exec"] for r in episode_results) / total_episodes
    avg_sr = sum(r["SR"] for r in episode_results) / total_episodes
    avg_tcr = sum(r["TCR"] for r in episode_results) / total_episodes
    avg_gcr = sum(r.get("GCR", 0) for r in episode_results) / total_episodes
    avg_plan_length = sum(r["PlanLength"] for r in episode_results) / total_episodes
    
    # Guard Trigger Distribution 집계
    guard_distribution = defaultdict(int)
    for result in episode_results:
        for guard, count in result.get("failed_guards", {}).items():
            guard_distribution[guard] += count
    
    return {
        "total_episodes": total_episodes,
        "avg_Exec": avg_exec,
        "avg_SR": avg_sr,
        "avg_TCR": avg_tcr,
        "avg_GCR": avg_gcr,
        "avg_PlanLength": avg_plan_length,
        "guard_trigger_distribution": dict(guard_distribution),
        "episode_results": episode_results
    }


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="완전 자동화된 평가 지표 수집 (또는 Baseline vs Physical Guard 비교)"
    )
    
    # 새로운 자동 평가 인터페이스
    parser.add_argument(
        "--execution-result",
        type=str,
        default=None,
        help="실행 결과 JSON 파일 경로 (단일 에피소드, 자동 평가용)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="평가 결과를 저장할 파일 경로 (선택사항)"
    )
    
    parser.add_argument(
        "--aggregate",
        type=str,
        default=None,
        help="여러 에피소드 결과를 집계할 디렉토리 경로 (선택사항)"
    )
    
    # 기존 비교 인터페이스 (호환성 유지)
    parser.add_argument(
        "--baseline-json",
        type=str,
        default=None,
        help="Baseline 결과 JSON 파일 경로 (비교 평가용)"
    )
    
    parser.add_argument(
        "--physical-guard-txt",
        type=str,
        default=None,
        help="Physical Guard 결과 텍스트 파일 경로 (비교 평가용)"
    )
    
    parser.add_argument(
        "--baseline-scene-graph",
        type=str,
        default=None,
        help="Baseline 업데이트된 Scene Graph JSON 파일 경로 (선택사항)"
    )
    
    parser.add_argument(
        "--physical-guard-scene-graph",
        type=str,
        default=None,
        help="Physical Guard 업데이트된 Scene Graph JSON 파일 경로 (선택사항)"
    )
    
    args = parser.parse_args()
    
    # 기존 비교 인터페이스 사용 (Baseline vs Physical Guard)
    if args.baseline_json and args.physical_guard_txt:
        # 기존 비교 함수들을 임시로 정의 (호환성 유지)
        def parse_program_to_actions_legacy(program_code: str) -> List[Dict[str, Any]]:
            """프로그램 코드를 파싱하여 액션 리스트로 변환 (기존 방식)"""
            lines = program_code.split("\n")
            plan = []
            for line in lines:
                line = line.strip()
                if line.startswith("assert") or line.startswith("else:") or line.startswith("#"):
                    continue
                match = re.match(r'(\w+)\(([^)]*)\)', line)
                if not match:
                    continue
                action = match.group(1)
                params = match.group(2)
                if not params:
                    continue
                params = [p.strip().strip("'\"") for p in params.split(",")]
                action_type = action
                if action == "GoTo":
                    action_type = "GoToObject"
                elif action == "Pickup":
                    action_type = "PickupObject"
                elif action == "Put":
                    action_type = "PutObject"
                elif action == "Open":
                    action_type = "OpenObject"
                elif action == "Close":
                    action_type = "CloseObject"
                if len(params) == 1:
                    plan.append({"type": action_type, "args": {"o": params[0]}, "line": line})
                elif len(params) == 2:
                    plan.append({"type": action_type, "args": {"o": params[0], "r": params[1]}, "line": line})
            return plan
        
        def extract_baseline_metrics_legacy(baseline_file: Path) -> Dict[str, Any]:
            """Baseline 결과 파일에서 지표 추출 (기존 방식)"""
            with open(baseline_file, "r", encoding="utf-8") as f:
                baseline_data = json.load(f)
            metrics = {
                "tasks": {},
                "total_tasks": len(baseline_data),
                "total_actions": 0,
                "total_passed_actions": 0,
                "total_failed_actions": 0,
                "total_recovery_actions": 0,
                "successful_tasks": 0,
                "plan_lengths": []  # Plan Length 추가
            }
            for task, program in baseline_data.items():
                actions = parse_program_to_actions_legacy(program)
                total_actions = len(actions)
                passed_actions = total_actions
                failed_actions = 0
                recovery_actions = 0
                task_success = None
                
                # Plan Length 계산
                plan_length = total_actions
                metrics["plan_lengths"].append(plan_length)
                
                task_metrics = {
                    "total_actions": total_actions,
                    "passed_actions": passed_actions,
                    "failed_actions": failed_actions,
                    "recovery_actions": recovery_actions,
                    "task_success": task_success,
                    "plan_length": plan_length
                }
                metrics["tasks"][task] = task_metrics
                metrics["total_actions"] += total_actions
                metrics["total_passed_actions"] += passed_actions
                metrics["total_failed_actions"] += failed_actions
                metrics["total_recovery_actions"] += recovery_actions
            return metrics
        
        def extract_physical_guard_metrics_legacy(physical_guard_file: Path) -> Dict[str, Any]:
            """Physical Guard 결과 파일에서 지표 추출 (기존 방식)"""
            with open(physical_guard_file, "r", encoding="utf-8") as f:
                content = f.read()
            metrics = {
                "tasks": {},
                "total_tasks": 0,
                "total_actions": 0,
                "total_passed_actions": 0,
                "total_failed_actions": 0,
                "total_recovery_actions": 0,
                "successful_tasks": 0,
                "guard_trigger_distribution": defaultdict(int),  # Guard Trigger Distribution 추가
                "plan_lengths": []  # Plan Length 추가
            }
            task_sections = content.split("Task: ")
            for section in task_sections[1:]:
                lines = section.split("\n")
                if not lines:
                    continue
                task = lines[0].strip()
                if not task:
                    continue
                metrics["total_tasks"] += 1
                total_actions = 0
                passed_actions = 0
                failed_actions = 0
                recovery_actions = 0
                task_success = None
                failed_guards_list = []  # 실패한 guard 수집
                in_summary = False
                in_failed_actions = False
                for i, line in enumerate(lines):
                    if "Physical Verification Summary:" in line:
                        in_summary = True
                        continue
                    if in_summary:
                        if "Total Actions:" in line:
                            match = re.search(r'Total Actions: (\d+)', line)
                            if match:
                                total_actions = int(match.group(1))
                        elif "Passed Actions:" in line:
                            match = re.search(r'Passed Actions: (\d+)', line)
                            if match:
                                passed_actions = int(match.group(1))
                        elif "Failed Actions:" in line:
                            match = re.search(r'Failed Actions: (\d+)', line)
                            if match:
                                failed_actions = int(match.group(1))
                            in_failed_actions = True
                            continue
                        elif in_failed_actions and line.strip().startswith("- "):
                            # Failed Actions 목록에서 guard 정보 추출
                            # 예: "  - GoToObject('Apple'): 1개 가드 실패: REACHABLE"
                            guard_match = re.search(r'가드 실패[:\s]+([^:]+)', line)
                            if guard_match:
                                guards_str = guard_match.group(1)
                                # 여러 guard가 있을 수 있음 (예: "REACHABLE, HOLDS")
                                guards = [g.strip() for g in guards_str.split(",")]
                                failed_guards_list.extend(guards)
                        elif "Task Completion Verification:" in line:
                            for j in range(i+1, min(i+5, len(lines))):
                                if "All tasks completed: Yes" in lines[j]:
                                    task_success = True
                                    metrics["successful_tasks"] += 1
                                    break
                                elif "All tasks completed: No" in lines[j]:
                                    task_success = False
                                    break
                            break
                
                # Guard Trigger Distribution 수집
                for guard in failed_guards_list:
                    metrics["guard_trigger_distribution"][guard] += 1
                
                # Plan Length 계산 (프로그램 코드에서 액션 수 계산)
                program_section = "\n".join(lines)
                # 주석처리되지 않은 액션 라인만 카운트
                action_lines = [l for l in program_section.split("\n") 
                                 if re.match(r'^\s*\w+\([^)]*\)', l.strip()) and not l.strip().startswith("#")]
                plan_length = len(action_lines)
                metrics["plan_lengths"].append(plan_length)
                
                program_section = "\n".join(lines)
                recovery_actions = program_section.count("[시스템 생성] 복구 액션")
                task_metrics = {
                    "total_actions": total_actions,
                    "passed_actions": passed_actions,
                    "failed_actions": failed_actions,
                    "recovery_actions": recovery_actions,
                    "task_success": task_success,
                    "plan_length": plan_length,
                    "failed_guards": failed_guards_list
                }
                metrics["tasks"][task] = task_metrics
                metrics["total_actions"] += total_actions
                metrics["total_passed_actions"] += passed_actions
                metrics["total_failed_actions"] += failed_actions
                metrics["total_recovery_actions"] += recovery_actions
            
            # Guard Trigger Distribution을 일반 dict로 변환
            metrics["guard_trigger_distribution"] = dict(metrics["guard_trigger_distribution"])
            return metrics
        
        def calculate_metrics_legacy(metrics: Dict[str, Any]) -> Dict[str, Any]:
            """지표 딕셔너리에서 평가 지표 계산 (기존 방식)"""
            total_tasks = metrics["total_tasks"]
            total_actions = metrics["total_actions"]
            total_passed_actions = metrics["total_passed_actions"]
            total_failed_actions = metrics["total_failed_actions"]
            total_recovery_actions = metrics["total_recovery_actions"]
            successful_tasks = metrics["successful_tasks"]
            tasks_with_result = sum(1 for t in metrics["tasks"].values() if t["task_success"] is not None)
            task_success_rate = (successful_tasks / tasks_with_result * 100) if tasks_with_result > 0 else None
            action_pass_rate = (total_passed_actions / total_actions * 100) if total_actions > 0 else 0.0
            recovery_effectiveness = (total_recovery_actions / total_failed_actions * 100) if total_failed_actions > 0 else 0.0
            plan_executability = ((total_actions - total_failed_actions) / total_actions * 100) if total_actions > 0 else 0.0
            
            # Plan Length 계산 (평균)
            plan_lengths = metrics.get("plan_lengths", [])
            avg_plan_length = sum(plan_lengths) / len(plan_lengths) if plan_lengths else 0.0
            
            # Guard Trigger Distribution
            guard_trigger_distribution = metrics.get("guard_trigger_distribution", {})
            
            return {
                "task_success_rate": task_success_rate,
                "action_pass_rate": action_pass_rate,
                "recovery_effectiveness": recovery_effectiveness,
                "plan_executability": plan_executability,
                "plan_length": avg_plan_length,  # Plan Length 추가
                "guard_trigger_distribution": guard_trigger_distribution,  # Guard Trigger Distribution 추가
                "total_tasks": total_tasks,
                "total_actions": total_actions,
                "total_passed_actions": total_passed_actions,
                "total_failed_actions": total_failed_actions,
                "total_recovery_actions": total_recovery_actions,
                "successful_tasks": successful_tasks,
                "tasks_with_result": tasks_with_result
            }
        
        def compare_results_legacy(baseline_metrics: Dict[str, Any], physical_guard_metrics: Dict[str, Any]) -> Dict[str, Any]:
            """Baseline과 Physical Guard 결과 비교 (기존 방식)"""
            comparison = {}
            metrics_to_compare = ["task_success_rate", "action_pass_rate", "recovery_effectiveness", "plan_executability", "plan_length"]
            for metric in metrics_to_compare:
                baseline_val = baseline_metrics.get(metric)
                pg_val = physical_guard_metrics.get(metric)
                if baseline_val is None or pg_val is None:
                    improvement = None
                    improvement_pct = None
                else:
                    improvement = pg_val - baseline_val
                    if baseline_val > 0:
                        improvement_pct = (improvement / baseline_val) * 100
                    else:
                        improvement_pct = float('inf') if improvement > 0 else 0.0
                comparison[metric] = {
                    "baseline": baseline_val,
                    "physical_guard": pg_val,
                    "improvement": improvement,
                    "improvement_pct": improvement_pct
                }
            
            # Guard Trigger Distribution 비교 (Baseline은 guard가 없으므로 Physical Guard만 표시)
            comparison["guard_trigger_distribution"] = {
                "baseline": {},
                "physical_guard": physical_guard_metrics.get("guard_trigger_distribution", {}),
                "improvement": None,
                "improvement_pct": None
            }
            
            return comparison
        
        def compare_scene_graphs_legacy(baseline_scene_graph_path: Path, physical_guard_scene_graph_path: Path) -> Dict[str, Any]:
            """Baseline과 Physical Guard의 Scene Graph를 비교 (기존 방식)"""
            comparison = {
                "baseline_objects_in_receptacles": set(),
                "physical_guard_objects_in_receptacles": set(),
                "missing_objects": [],
                "extra_objects": [],
                "all_tasks_completed": True
            }
            try:
                with open(baseline_scene_graph_path, "r", encoding="utf-8") as f:
                    baseline_sg = json.load(f)
                with open(physical_guard_scene_graph_path, "r", encoding="utf-8") as f:
                    pg_sg = json.load(f)
                baseline_edges = baseline_sg.get("edges", [])
                pg_edges = pg_sg.get("edges", [])
                for edge in baseline_edges:
                    if edge.get("edgeType") == "IN":
                        source = edge.get("source")
                        target = edge.get("target")
                        if source and target:
                            comparison["baseline_objects_in_receptacles"].add((source, target))
                for edge in pg_edges:
                    if edge.get("edgeType") == "IN":
                        source = edge.get("source")
                        target = edge.get("target")
                        if source and target:
                            comparison["physical_guard_objects_in_receptacles"].add((source, target))
                missing = comparison["baseline_objects_in_receptacles"] - comparison["physical_guard_objects_in_receptacles"]
                comparison["missing_objects"] = list(missing)
                extra = comparison["physical_guard_objects_in_receptacles"] - comparison["baseline_objects_in_receptacles"]
                comparison["extra_objects"] = list(extra)
                comparison["all_tasks_completed"] = len(comparison["missing_objects"]) == 0
            except Exception as e:
                print(f"⚠️  Scene Graph 비교 중 오류 발생: {e}")
                comparison["error"] = str(e)
            return comparison
        
        def print_comparison_report_legacy(baseline_metrics: Dict[str, Any], physical_guard_metrics: Dict[str, Any], comparison: Dict[str, Any]):
            """비교 결과를 출력 (기존 방식)"""
            print("\n" + "=" * 100)
            print("평가 결과 비교: Baseline(ProgPrompt) vs Physical Guard")
            print("=" * 100)
            
            # 기존 지표들
            for metric_name in ["task_success_rate", "action_pass_rate", "recovery_effectiveness", "plan_executability"]:
                metric_data = comparison[metric_name]
                print(f"\n[{metric_name.replace('_', ' ').title()}]")
                print("-" * 100)
                baseline_val = metric_data["baseline"]
                pg_val = metric_data["physical_guard"]
                improvement = metric_data["improvement"]
                improvement_pct = metric_data["improvement_pct"]
                if baseline_val is None:
                    print(f"  Baseline:        N/A (물리적 검증 없음)")
                else:
                    print(f"  Baseline:        {baseline_val:.2f}%")
                if pg_val is None:
                    print(f"  Physical Guard:  N/A")
                else:
                    print(f"  Physical Guard:  {pg_val:.2f}%")
                if improvement is not None:
                    print(f"  개선:            {improvement:+.2f}% ({improvement_pct:+.2f}%)" if improvement != 0 else f"  개선:            {improvement:.2f}% (변화 없음)")
            
            # Plan Length 비교
            if "plan_length" in comparison:
                metric_data = comparison["plan_length"]
                print(f"\n[Plan Length (평균 액션 수)]")
                print("-" * 100)
                baseline_val = metric_data["baseline"]
                pg_val = metric_data["physical_guard"]
                improvement = metric_data["improvement"]
                improvement_pct = metric_data["improvement_pct"]
                print(f"  Baseline:        {baseline_val:.2f}")
                print(f"  Physical Guard:  {pg_val:.2f}")
                if improvement is not None:
                    print(f"  차이:            {improvement:+.2f} ({improvement_pct:+.2f}%)" if improvement != 0 else f"  차이:            {improvement:.2f} (변화 없음)")
            
            # Guard Trigger Distribution
            if "guard_trigger_distribution" in comparison:
                guard_data = comparison["guard_trigger_distribution"]
                print(f"\n[Guard Trigger Distribution]")
                print("-" * 100)
                print(f"  Baseline:        N/A (guard 검증 없음)")
                pg_guards = guard_data["physical_guard"]
                if pg_guards:
                    print(f"  Physical Guard:")
                    for guard, count in sorted(pg_guards.items(), key=lambda x: x[1], reverse=True):
                        print(f"    {guard}: {count}회")
                else:
                    print(f"  Physical Guard:  실패한 guard 없음")
            
            print("\n" + "=" * 100)
            print("\n⚠️  참고: Exec, SR, TCR은 실행 결과가 필요하므로 계획 생성 단계에서는 계산할 수 없습니다.")
            print("         실행 결과가 있으면 --execution-result 옵션을 사용하여 자동 평가를 수행하세요.")
            print("=" * 100)
        
        # 기존 비교 함수 사용
        
        baseline_path = Path(args.baseline_json)
        physical_guard_path = Path(args.physical_guard_txt)
        
        if not baseline_path.exists():
            print(f"❌ 오류: Baseline 파일을 찾을 수 없습니다: {baseline_path}")
            return
        
        if not physical_guard_path.exists():
            print(f"❌ 오류: Physical Guard 파일을 찾을 수 없습니다: {physical_guard_path}")
            return
        
        print(f"📊 Baseline 결과 로드 중: {baseline_path}")
        baseline_raw_metrics = extract_baseline_metrics_legacy(baseline_path)
        baseline_metrics = calculate_metrics_legacy(baseline_raw_metrics)
        
        print(f"📊 Physical Guard 결과 로드 중: {physical_guard_path}")
        physical_guard_raw_metrics = extract_physical_guard_metrics_legacy(physical_guard_path)
        physical_guard_metrics = calculate_metrics_legacy(physical_guard_raw_metrics)
        
        # 비교 결과 계산
        comparison = compare_results_legacy(baseline_metrics, physical_guard_metrics)
        
        # Scene Graph 비교 (선택사항)
        scene_graph_comparison = None
        if args.baseline_scene_graph and args.physical_guard_scene_graph:
            baseline_sg_path = Path(args.baseline_scene_graph)
            pg_sg_path = Path(args.physical_guard_scene_graph)
            
            if baseline_sg_path.exists() and pg_sg_path.exists():
                print(f"\n📊 Scene Graph 비교 중...")
                print(f"  Baseline Scene Graph: {baseline_sg_path}")
                print(f"  Physical Guard Scene Graph: {pg_sg_path}")
                
                scene_graph_comparison = compare_scene_graphs_legacy(baseline_sg_path, pg_sg_path)
                
                if "error" not in scene_graph_comparison:
                    print(f"\n[Scene Graph 비교 결과]")
                    print("-" * 100)
                    print(f"  Baseline 객체-수용체 관계: {len(scene_graph_comparison['baseline_objects_in_receptacles'])}개")
                    print(f"  Physical Guard 객체-수용체 관계: {len(scene_graph_comparison['physical_guard_objects_in_receptacles'])}개")
                    print(f"  누락된 객체: {len(scene_graph_comparison['missing_objects'])}개")
                    if scene_graph_comparison['missing_objects']:
                        for obj_id, recp_id in scene_graph_comparison['missing_objects']:
                            print(f"    - {obj_id} → {recp_id}")
                    print(f"  추가된 객체: {len(scene_graph_comparison['extra_objects'])}개")
                    if scene_graph_comparison['extra_objects']:
                        for obj_id, recp_id in scene_graph_comparison['extra_objects']:
                            print(f"    - {obj_id} → {recp_id}")
                    print(f"  모든 task 완료: {'✅ Yes' if scene_graph_comparison['all_tasks_completed'] else '❌ No'}")
            else:
                print(f"⚠️  Scene Graph 파일을 찾을 수 없습니다.")
                if not baseline_sg_path.exists():
                    print(f"  Baseline: {baseline_sg_path}")
                if not pg_sg_path.exists():
                    print(f"  Physical Guard: {pg_sg_path}")
        
        # 결과 출력
        print_comparison_report_legacy(baseline_metrics, physical_guard_metrics, comparison)
        
        # 결과를 파일로 저장 (선택사항)
        if args.output:
            output_path = Path(args.output)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("=" * 100 + "\n")
                f.write("평가 결과 비교: Baseline(ProgPrompt) vs Physical Guard\n")
                f.write("=" * 100 + "\n\n")
                
                f.write(f"평가 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write(f"Baseline 파일: {baseline_path}\n")
                f.write(f"Physical Guard 파일: {physical_guard_path}\n\n")
                
                # 각 지표별 상세 정보
                f.write("[지표 상세]\n")
                f.write("-" * 100 + "\n")
                
                for metric_name, metric_data in comparison.items():
                    f.write(f"\n{metric_name}:\n")
                    f.write(f"  Baseline:        {metric_data['baseline']}\n")
                    f.write(f"  Physical Guard:  {metric_data['physical_guard']}\n")
                    f.write(f"  Improvement:     {metric_data['improvement']}\n")
                    f.write(f"  Improvement %:   {metric_data['improvement_pct']}\n")
                
                # Scene Graph 비교 결과 추가
                if scene_graph_comparison and "error" not in scene_graph_comparison:
                    f.write("\n[Scene Graph 비교 결과]\n")
                    f.write("-" * 100 + "\n")
                    f.write(f"  Baseline 객체-수용체 관계: {len(scene_graph_comparison['baseline_objects_in_receptacles'])}개\n")
                    f.write(f"  Physical Guard 객체-수용체 관계: {len(scene_graph_comparison['physical_guard_objects_in_receptacles'])}개\n")
                    f.write(f"  누락된 객체: {len(scene_graph_comparison['missing_objects'])}개\n")
                    if scene_graph_comparison['missing_objects']:
                        for obj_id, recp_id in scene_graph_comparison['missing_objects']:
                            f.write(f"    - {obj_id} → {recp_id}\n")
                    f.write(f"  추가된 객체: {len(scene_graph_comparison['extra_objects'])}개\n")
                    if scene_graph_comparison['extra_objects']:
                        for obj_id, recp_id in scene_graph_comparison['extra_objects']:
                            f.write(f"    - {obj_id} → {recp_id}\n")
                    f.write(f"  모든 task 완료: {'Yes' if scene_graph_comparison['all_tasks_completed'] else 'No'}\n")
                
                f.write("\n" + "=" * 100 + "\n")
            
            print(f"\n✅ 비교 결과가 {output_path}에 저장되었습니다.")
        
        return
    
    # 새로운 자동 평가 인터페이스 사용
    if args.aggregate:
        # 여러 에피소드 결과 집계
        aggregate_dir = Path(args.aggregate)
        if not aggregate_dir.exists():
            print(f"❌ 오류: 집계 디렉토리를 찾을 수 없습니다: {aggregate_dir}")
            return
        
        # 디렉토리 내 모든 평가 결과 파일 찾기
        evaluation_files = list(aggregate_dir.glob("*_evaluation.json"))
        
        if not evaluation_files:
            print(f"⚠️  평가 결과 파일을 찾을 수 없습니다: {aggregate_dir}")
            return
        
        print(f"📊 {len(evaluation_files)}개의 평가 결과 파일 로드 중...")
        episode_results = []
        for eval_file in evaluation_files:
            with open(eval_file, "r", encoding="utf-8") as f:
                episode_results.append(json.load(f))
        
        # 집계
        aggregated = aggregate_episode_results(episode_results)
        
        # 출력
        output_path = Path(args.output) if args.output else aggregate_dir / "aggregated_evaluation.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(aggregated, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 집계 결과가 {output_path}에 저장되었습니다.")
        print(f"\n집계 결과:")
        print(f"  총 에피소드: {aggregated['total_episodes']}")
        print(f"  평균 Exec: {aggregated['avg_Exec']:.2f}")
        print(f"  평균 SR: {aggregated['avg_SR']:.2f}%")
        print(f"  평균 TCR: {aggregated['avg_TCR']:.2f}%")
        print(f"  평균 GCR: {aggregated['avg_GCR']:.2f}%")
        print(f"  평균 Plan Length: {aggregated['avg_PlanLength']:.2f}")
        print(f"\nGuard Trigger Distribution:")
        for guard, count in sorted(aggregated['guard_trigger_distribution'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {guard}: {count}회")
    
    else:
        # 단일 에피소드 평가
        if not args.execution_result:
            parser.print_help()
            return

        execution_result_path = Path(args.execution_result)
        if not execution_result_path.exists():
            print(f"❌ 오류: 실행 결과 파일을 찾을 수 없습니다: {execution_result_path}")
            return
        
        print(f"📊 실행 결과 로드 중: {execution_result_path}")
        execution_result = load_execution_result(execution_result_path)
        
        # 평가 지표 계산
        episode_id = execution_result.get("episode_id", "unknown")
        instruction = execution_result.get("instruction", "")
        executed_actions = execution_result.get("executed_actions", [])
        final_metadata = execution_result.get("final_metadata")
        final_scene_graph = execution_result.get("final_scene_graph")
        failed_guards = execution_result.get("failed_guards", [])
        
        evaluation_result = evaluate_episode(
            episode_id=episode_id,
            instruction=instruction,
            executed_actions=executed_actions,
            final_metadata=final_metadata,
            final_scene_graph=final_scene_graph,
            failed_guards=failed_guards
        )
        
        # 결과 출력
        output_path = Path(args.output) if args.output else Path(f"{episode_id}_evaluation.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(evaluation_result, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 평가 결과가 {output_path}에 저장되었습니다.")
        print(f"\n평가 지표:")
        print(f"  Episode ID: {evaluation_result['episode_id']}")
        print(f"  Exec: {evaluation_result['Exec']}")
        print(f"  SR: {evaluation_result['SR']:.2f}%")
        print(f"  TCR: {evaluation_result['TCR']:.2f}%")
        print(f"  GCR: {evaluation_result.get('GCR', 0):.2f}%")
        print(f"  Plan Length: {evaluation_result['PlanLength']}")
        if evaluation_result['failed_guards']:
            print(f"\n  Guard Trigger Distribution:")
            for guard, count in sorted(evaluation_result['failed_guards'].items(), key=lambda x: x[1], reverse=True):
                print(f"    {guard}: {count}회")
        if evaluation_result['missing_tasks']:
            print(f"\n  Missing Tasks:")
            for task in evaluation_result['missing_tasks']:
                print(f"    {task['type']}({task['args']})")


if __name__ == "__main__":
    main()