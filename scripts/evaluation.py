#!/usr/bin/env python3
"""
Baseline(ProgPrompt)와 Physical Guard 결과를 비교하는 평가 스크립트

평가 지표:
1. Task Success Rate (작업 성공률)
2. Action Pass Rate (액션 통과율)
3. Recovery Action Effectiveness (복구 액션 효과성)
4. Plan Executability (계획 실행 가능성)
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime


def parse_program_to_actions(program_code: str) -> List[Dict[str, Any]]:
    """
    프로그램 코드를 파싱하여 액션 리스트로 변환
    
    Args:
        program_code: 프로그램 코드 (def 형태)
        
    Returns:
        액션 리스트 [{"type": str, "args": dict, "line": str}, ...]
    """
    lines = program_code.split("\n")
    plan = []
    
    for line in lines:
        line = line.strip()
        
        # assert, else:, 주석 제거
        if line.startswith("assert") or line.startswith("else:") or line.startswith("#"):
            continue
        
        # 액션 파싱
        match = re.match(r'(\w+)\(([^)]*)\)', line)
        if not match:
            continue
        
        action = match.group(1)
        params = match.group(2)
        
        if not params:
            continue
        
        params = [p.strip().strip("'\"") for p in params.split(",")]
        
        # 액션 타입 정규화
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
            plan.append({
                "type": action_type,
                "args": {"o": params[0]},
                "line": line
            })
        elif len(params) == 2:
            plan.append({
                "type": action_type,
                "args": {"o": params[0], "r": params[1]},
                "line": line
            })
    
    return plan


def extract_baseline_metrics(baseline_file: Path) -> Dict[str, Any]:
    """
    Baseline 결과 파일에서 지표 추출
    
    Args:
        baseline_file: Baseline 결과 JSON 파일 경로
        
    Returns:
        지표 딕셔너리
    """
    with open(baseline_file, "r", encoding="utf-8") as f:
        baseline_data = json.load(f)
    
    metrics = {
        "tasks": {},
        "total_tasks": len(baseline_data),
        "total_actions": 0,
        "total_passed_actions": 0,
        "total_failed_actions": 0,
        "total_recovery_actions": 0,
        "successful_tasks": 0
    }
    
    for task, program in baseline_data.items():
        # 프로그램 파싱하여 액션 수 계산
        actions = parse_program_to_actions(program)
        total_actions = len(actions)
        
        # Baseline은 물리적 검증을 하지 않으므로 모든 액션이 통과했다고 가정
        passed_actions = total_actions
        failed_actions = 0
        recovery_actions = 0
        
        # Baseline은 물리적 검증이 없으므로 작업 성공 여부를 알 수 없음
        # 프로그램이 생성되었다는 것만으로는 성공 여부를 판단할 수 없으므로 None으로 설정
        task_success = None
        
        task_metrics = {
            "total_actions": total_actions,
            "passed_actions": passed_actions,
            "failed_actions": failed_actions,
            "recovery_actions": recovery_actions,
            "task_success": task_success
        }
        
        metrics["tasks"][task] = task_metrics
        metrics["total_actions"] += total_actions
        metrics["total_passed_actions"] += passed_actions
        metrics["total_failed_actions"] += failed_actions
        metrics["total_recovery_actions"] += recovery_actions
    
    return metrics


def extract_physical_guard_metrics(physical_guard_file: Path) -> Dict[str, Any]:
    """
    Physical Guard 결과 파일에서 지표 추출
    
    Args:
        physical_guard_file: Physical Guard 결과 텍스트 파일 경로
        
    Returns:
        지표 딕셔너리
    """
    with open(physical_guard_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    metrics = {
        "tasks": {},
        "total_tasks": 0,
        "total_actions": 0,
        "total_passed_actions": 0,
        "total_failed_actions": 0,
        "total_recovery_actions": 0,
        "successful_tasks": 0
    }
    
    # Task별로 섹션 분리
    task_sections = content.split("Task: ")
    
    for section in task_sections[1:]:  # 첫 번째는 빈 문자열
        lines = section.split("\n")
        if not lines:
            continue
        
        task = lines[0].strip()
        if not task:
            continue
        
        metrics["total_tasks"] += 1
        
        # Physical Verification Summary 찾기
        total_actions = 0
        passed_actions = 0
        failed_actions = 0
        recovery_actions = 0
        task_success = None
        
        in_summary = False
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
                elif "Task Completion Verification:" in line:
                    # 다음 몇 줄을 확인하여 all_completed 여부 확인
                    for j in range(i+1, min(i+5, len(lines))):
                        if "All tasks completed: Yes" in lines[j]:
                            task_success = True
                            metrics["successful_tasks"] += 1
                            break
                        elif "All tasks completed: No" in lines[j]:
                            task_success = False
                            break
                    break
        
        # 복구 액션 수 계산 (프로그램 코드에서 [시스템 생성] 복구 액션 찾기)
        program_section = "\n".join(lines)
        recovery_actions = program_section.count("[시스템 생성] 복구 액션")
        
        task_metrics = {
            "total_actions": total_actions,
            "passed_actions": passed_actions,
            "failed_actions": failed_actions,
            "recovery_actions": recovery_actions,
            "task_success": task_success
        }
        
        metrics["tasks"][task] = task_metrics
        metrics["total_actions"] += total_actions
        metrics["total_passed_actions"] += passed_actions
        metrics["total_failed_actions"] += failed_actions
        metrics["total_recovery_actions"] += recovery_actions
    
    return metrics


def calculate_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    """
    지표 딕셔너리에서 평가 지표 계산
    
    Args:
        metrics: 지표 딕셔너리
        
    Returns:
        계산된 평가 지표 딕셔너리
    """
    total_tasks = metrics["total_tasks"]
    total_actions = metrics["total_actions"]
    total_passed_actions = metrics["total_passed_actions"]
    total_failed_actions = metrics["total_failed_actions"]
    total_recovery_actions = metrics["total_recovery_actions"]
    successful_tasks = metrics["successful_tasks"]
    
    # 1. Task Success Rate (작업 성공률)
    # task_success가 None인 경우는 제외
    tasks_with_result = sum(1 for t in metrics["tasks"].values() if t["task_success"] is not None)
    if tasks_with_result > 0:
        task_success_rate = (successful_tasks / tasks_with_result) * 100
    else:
        task_success_rate = None
    
    # 2. Action Pass Rate (액션 통과율)
    if total_actions > 0:
        action_pass_rate = (total_passed_actions / total_actions) * 100
    else:
        action_pass_rate = 0.0
    
    # 3. Recovery Action Effectiveness (복구 액션 효과성)
    if total_failed_actions > 0:
        recovery_effectiveness = (total_recovery_actions / total_failed_actions) * 100
    else:
        recovery_effectiveness = 0.0  # 실패한 액션이 없으면 복구 효과성은 0
    
    # 4. Plan Executability (계획 실행 가능성)
    if total_actions > 0:
        plan_executability = ((total_actions - total_failed_actions) / total_actions) * 100
    else:
        plan_executability = 0.0
    
    return {
        "task_success_rate": task_success_rate,
        "action_pass_rate": action_pass_rate,
        "recovery_effectiveness": recovery_effectiveness,
        "plan_executability": plan_executability,
        "total_tasks": total_tasks,
        "total_actions": total_actions,
        "total_passed_actions": total_passed_actions,
        "total_failed_actions": total_failed_actions,
        "total_recovery_actions": total_recovery_actions,
        "successful_tasks": successful_tasks,
        "tasks_with_result": tasks_with_result
    }


def compare_scene_graphs(
    baseline_scene_graph_path: Path,
    physical_guard_scene_graph_path: Path
) -> Dict[str, Any]:
    """
    Baseline과 Physical Guard의 Scene Graph를 비교하여 누락된 task 확인
    
    Args:
        baseline_scene_graph_path: Baseline 업데이트된 Scene Graph 파일 경로
        physical_guard_scene_graph_path: Physical Guard 업데이트된 Scene Graph 파일 경로
        
    Returns:
        비교 결과 딕셔너리
    """
    comparison = {
        "baseline_objects_in_receptacles": set(),
        "physical_guard_objects_in_receptacles": set(),
        "missing_objects": [],
        "extra_objects": [],
        "all_tasks_completed": True
    }
    
    try:
        # Baseline Scene Graph 로드
        with open(baseline_scene_graph_path, "r", encoding="utf-8") as f:
            baseline_sg = json.load(f)
        
        # Physical Guard Scene Graph 로드
        with open(physical_guard_scene_graph_path, "r", encoding="utf-8") as f:
            pg_sg = json.load(f)
        
        # IN 엣지에서 객체-수용체 관계 추출
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
        
        # 누락된 객체 확인 (Baseline에는 있지만 Physical Guard에는 없는 경우)
        missing = comparison["baseline_objects_in_receptacles"] - comparison["physical_guard_objects_in_receptacles"]
        comparison["missing_objects"] = list(missing)
        
        # 추가된 객체 확인 (Physical Guard에는 있지만 Baseline에는 없는 경우)
        extra = comparison["physical_guard_objects_in_receptacles"] - comparison["baseline_objects_in_receptacles"]
        comparison["extra_objects"] = list(extra)
        
        # 모든 task가 완료되었는지 확인 (누락된 객체가 없으면 완료)
        comparison["all_tasks_completed"] = len(comparison["missing_objects"]) == 0
        
    except FileNotFoundError as e:
        print(f"⚠️  Scene Graph 파일을 찾을 수 없습니다: {e}")
        comparison["error"] = str(e)
    except Exception as e:
        print(f"⚠️  Scene Graph 비교 중 오류 발생: {e}")
        comparison["error"] = str(e)
    
    return comparison


def compare_results(baseline_metrics: Dict[str, float], physical_guard_metrics: Dict[str, float]) -> Dict[str, Any]:
    """
    Baseline과 Physical Guard 결과 비교
    
    Args:
        baseline_metrics: Baseline 지표
        physical_guard_metrics: Physical Guard 지표
        
    Returns:
        비교 결과 딕셔너리
    """
    comparison = {}
    
    # 각 지표별 비교
    metrics_to_compare = [
        "task_success_rate",
        "action_pass_rate",
        "recovery_effectiveness",
        "plan_executability"
    ]
    
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
    
    return comparison


def print_comparison_report(baseline_metrics: Dict[str, float], 
                           physical_guard_metrics: Dict[str, float],
                           comparison: Dict[str, Any]):
    """
    비교 결과를 출력
    
    Args:
        baseline_metrics: Baseline 지표
        physical_guard_metrics: Physical Guard 지표
        comparison: 비교 결과
    """
    print("\n" + "=" * 100)
    print("평가 결과 비교: Baseline(ProgPrompt) vs Physical Guard")
    print("=" * 100)
    
    print("\n[1] Task Success Rate (작업 성공률)")
    print("-" * 100)
    baseline_val = comparison["task_success_rate"]["baseline"]
    pg_val = comparison["task_success_rate"]["physical_guard"]
    improvement = comparison["task_success_rate"]["improvement"]
    improvement_pct = comparison["task_success_rate"]["improvement_pct"]
    
    if baseline_val is None:
        print(f"  Baseline:        N/A (물리적 검증 없음)")
    else:
        print(f"  Baseline:        {baseline_val:.2f}%")
    
    if pg_val is None:
        print(f"  Physical Guard:  N/A")
    else:
        print(f"  Physical Guard:  {pg_val:.2f}%")
    
    if improvement is not None:
        if improvement > 0:
            print(f"  개선:            +{improvement:.2f}% ({improvement_pct:+.2f}%)")
        elif improvement < 0:
            print(f"  개선:            {improvement:.2f}% ({improvement_pct:.2f}%)")
        else:
            print(f"  개선:            {improvement:.2f}% (변화 없음)")
    else:
        print(f"  개선:            N/A")
    
    print(f"\n  상세:")
    print(f"    Baseline:        {baseline_metrics.get('successful_tasks', 0)}/{baseline_metrics.get('tasks_with_result', 0)} tasks")
    print(f"    Physical Guard:  {physical_guard_metrics.get('successful_tasks', 0)}/{physical_guard_metrics.get('tasks_with_result', 0)} tasks")
    
    print("\n[2] Action Pass Rate (액션 통과율)")
    print("-" * 100)
    baseline_val = comparison["action_pass_rate"]["baseline"]
    pg_val = comparison["action_pass_rate"]["physical_guard"]
    improvement = comparison["action_pass_rate"]["improvement"]
    improvement_pct = comparison["action_pass_rate"]["improvement_pct"]
    
    print(f"  Baseline:        {baseline_val:.2f}%")
    print(f"  Physical Guard:  {pg_val:.2f}%")
    
    if improvement > 0:
        print(f"  개선:            +{improvement:.2f}% ({improvement_pct:+.2f}%)")
    elif improvement < 0:
        print(f"  개선:            {improvement:.2f}% ({improvement_pct:.2f}%)")
    else:
        print(f"  개선:            {improvement:.2f}% (변화 없음)")
    
    print(f"\n  상세:")
    print(f"    Baseline:        {baseline_metrics.get('total_passed_actions', 0)}/{baseline_metrics.get('total_actions', 0)} actions")
    print(f"    Physical Guard:  {physical_guard_metrics.get('total_passed_actions', 0)}/{physical_guard_metrics.get('total_actions', 0)} actions")
    
    print("\n[3] Recovery Action Effectiveness (복구 액션 효과성)")
    print("-" * 100)
    baseline_val = comparison["recovery_effectiveness"]["baseline"]
    pg_val = comparison["recovery_effectiveness"]["physical_guard"]
    improvement = comparison["recovery_effectiveness"]["improvement"]
    improvement_pct = comparison["recovery_effectiveness"]["improvement_pct"]
    
    print(f"  Baseline:        {baseline_val:.2f}% (복구 액션 없음)")
    print(f"  Physical Guard:  {pg_val:.2f}%")
    
    if improvement > 0:
        print(f"  개선:            +{improvement:.2f}% ({improvement_pct:+.2f}%)")
    elif improvement < 0:
        print(f"  개선:            {improvement:.2f}% ({improvement_pct:.2f}%)")
    else:
        print(f"  개선:            {improvement:.2f}% (변화 없음)")
    
    print(f"\n  상세:")
    print(f"    Baseline:        {baseline_metrics.get('total_recovery_actions', 0)} recovery actions")
    print(f"    Physical Guard:  {physical_guard_metrics.get('total_recovery_actions', 0)} recovery actions")
    print(f"    Physical Guard:  {physical_guard_metrics.get('total_failed_actions', 0)} failed actions")
    
    print("\n[4] Plan Executability (계획 실행 가능성)")
    print("-" * 100)
    baseline_val = comparison["plan_executability"]["baseline"]
    pg_val = comparison["plan_executability"]["physical_guard"]
    improvement = comparison["plan_executability"]["improvement"]
    improvement_pct = comparison["plan_executability"]["improvement_pct"]
    
    print(f"  Baseline:        {baseline_val:.2f}%")
    print(f"  Physical Guard:  {pg_val:.2f}%")
    
    if improvement > 0:
        print(f"  개선:            +{improvement:.2f}% ({improvement_pct:+.2f}%)")
    elif improvement < 0:
        print(f"  개선:            {improvement:.2f}% ({improvement_pct:.2f}%)")
    else:
        print(f"  개선:            {improvement:.2f}% (변화 없음)")
    
    print(f"\n  상세:")
    print(f"    Baseline:        {baseline_metrics.get('total_actions', 0) - baseline_metrics.get('total_failed_actions', 0)}/{baseline_metrics.get('total_actions', 0)} executable actions")
    print(f"    Physical Guard:  {physical_guard_metrics.get('total_actions', 0) - physical_guard_metrics.get('total_failed_actions', 0)}/{physical_guard_metrics.get('total_actions', 0)} executable actions")
    
    print("\n" + "=" * 100)


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="Baseline(ProgPrompt)와 Physical Guard 결과 비교 평가"
    )
    
    parser.add_argument(
        "--baseline-json",
        type=str,
        required=True,
        help="Baseline 결과 JSON 파일 경로"
    )
    
    parser.add_argument(
        "--physical-guard-txt",
        type=str,
        required=True,
        help="Physical Guard 결과 텍스트 파일 경로"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="비교 결과를 저장할 파일 경로 (선택사항)"
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
    
    # 파일 경로 확인
    baseline_path = Path(args.baseline_json)
    physical_guard_path = Path(args.physical_guard_txt)
    
    if not baseline_path.exists():
        print(f"❌ 오류: Baseline 파일을 찾을 수 없습니다: {baseline_path}")
        return
    
    if not physical_guard_path.exists():
        print(f"❌ 오류: Physical Guard 파일을 찾을 수 없습니다: {physical_guard_path}")
        return
    
    print(f"📊 Baseline 결과 로드 중: {baseline_path}")
    baseline_raw_metrics = extract_baseline_metrics(baseline_path)
    baseline_metrics = calculate_metrics(baseline_raw_metrics)
    
    print(f"📊 Physical Guard 결과 로드 중: {physical_guard_path}")
    physical_guard_raw_metrics = extract_physical_guard_metrics(physical_guard_path)
    physical_guard_metrics = calculate_metrics(physical_guard_raw_metrics)
    
    # 비교 결과 계산
    comparison = compare_results(baseline_metrics, physical_guard_metrics)
    
    # Scene Graph 비교 (선택사항)
    scene_graph_comparison = None
    if args.baseline_scene_graph and args.physical_guard_scene_graph:
        baseline_sg_path = Path(args.baseline_scene_graph)
        pg_sg_path = Path(args.physical_guard_scene_graph)
        
        if baseline_sg_path.exists() and pg_sg_path.exists():
            print(f"\n📊 Scene Graph 비교 중...")
            print(f"  Baseline Scene Graph: {baseline_sg_path}")
            print(f"  Physical Guard Scene Graph: {pg_sg_path}")
            
            scene_graph_comparison = compare_scene_graphs(baseline_sg_path, pg_sg_path)
            
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
    print_comparison_report(baseline_metrics, physical_guard_metrics, comparison)
    
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


if __name__ == "__main__":
    main()

