#!/usr/bin/env python3
"""
각 FloorPlan을 10번씩 실행하여 평균 GCR을 계산하고 Excel 파일로 저장하는 스크립트

실행 순서:
1. FloorPlan1.json, FloorPlan216.json, FloorPlan325.json, FloorPlan403.json 각각 10번 실행
2. 각 task마다 10번의 시도 평균 GCR 기록
3. 각 Scene마다 평균 GCR을 Baseline과 비교
4. result_half_test.xlsx 파일에 저장
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict

# Excel 파일 생성을 위한 라이브러리
try:
    import openpyxl
    from openpyxl.styles import Font, Alignment, PatternFill
    from openpyxl.utils import get_column_letter
except ImportError:
    print("❌ openpyxl이 설치되지 않았습니다. 설치 중...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl"])
    import openpyxl
    from openpyxl.styles import Font, Alignment, PatternFill
    from openpyxl.utils import get_column_letter

# evaluate_results.py의 함수들을 import하기 위해 경로 추가
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from evaluate_results import (
    parse_json_plan_file,
    load_expected_results,
    execute_and_evaluate_task
)
from ai2thor_connector_ithor import AI2ThorExecutor

def get_folder_from_fp_number(fp_number: int) -> str:
    """FloorPlan 번호에 따라 폴더 이름을 반환"""
    if fp_number in [1, 2]:
        return "Kitchen"
    elif fp_number in [216, 224]:
        return "LivingRoom"
    elif fp_number in [325, 326]:
        return "BedRoom"
    elif fp_number in [403, 425]:
        return "BathRoom"
    else:
        return "Kitchen"

def find_result_files(fp_number: int, folder: str) -> tuple[str, str]:
    """
    Physical Guard와 Baseline 결과 JSON 파일을 찾습니다.
    
    Returns:
        (physical_guard_file, baseline_file) 경로 튜플
    """
    import glob
    
    script_dir = Path(__file__).parent
    fp_folder = f"FP{fp_number}"
    
    # Physical Guard 파일 찾기
    patterns = [
        script_dir.parent / "results" / folder / fp_folder / "physical_guard_result_*.json",
        script_dir.parent / "results" / folder / "physical_guard_result_*.json",
        script_dir.parent / "results" / "physical_guard_result_*.json"
    ]
    
    physical_guard_file = None
    for pattern in patterns:
        files = glob.glob(str(pattern))
        if files:
            physical_guard_file = max(files, key=os.path.getmtime)
            break
    
    # Baseline 파일 찾기
    patterns = [
        script_dir.parent / "results" / folder / fp_folder / "baseline_result_*.json",
        script_dir.parent / "results" / folder / "baseline_result_*.json",
        script_dir.parent / "results" / "baseline_result_*.json"
    ]
    
    baseline_file = None
    for pattern in patterns:
        files = glob.glob(str(pattern))
        if files:
            baseline_file = max(files, key=os.path.getmtime)
            break
    
    return physical_guard_file, baseline_file

def run_physical_guard_subprocess(fp_number: int, task_file: str) -> bool:
    """physical_guard.py를 subprocess로 실행"""
    script_dir = Path(__file__).parent
    physical_guard_script = script_dir / "physical_guard.py"
    
    cmd = [sys.executable, str(physical_guard_script), "--scene-number", str(fp_number)]
    if task_file:
        cmd.extend(["--task-file", task_file])
    
    try:
        result = subprocess.run(
            cmd,
            cwd=script_dir.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False
        )
        return result.returncode == 0
    except Exception as e:
        print(f"❌ physical_guard.py 실행 실패: {e}")
        return False

def run_baseline_subprocess(fp_number: int, task_file: str) -> bool:
    """Baseline(ProgPrompt).py를 subprocess로 실행"""
    script_dir = Path(__file__).parent
    baseline_script = script_dir / "Baseline(ProgPrompt).py"
    
    if not baseline_script.exists():
        print(f"⚠️  Baseline 스크립트를 찾을 수 없습니다: {baseline_script}")
        return False
    
    cmd = [sys.executable, str(baseline_script), "--scene-number", str(fp_number)]
    if task_file:
        cmd.extend(["--task-file", task_file])
    
    try:
        result = subprocess.run(
            cmd,
            cwd=script_dir.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False
        )
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Baseline 실행 실패: {e}")
        return False

def run_single_evaluation(
    fp_number: int,
    folder: str,
    test_file: str,
    run_number: int
) -> Dict[str, Any]:
    """
    단일 평가 실행
    - physical_guard.py 실행
    - Baseline(ProgPrompt).py 실행
    - evaluate_results.py 로직으로 평가
    
    Returns:
        {
            "physical_guard": [{"task": str, "gcr": float, ...}, ...],
            "baseline": [{"task": str, "gcr": float, ...}, ...]
        }
    """
    print(f"\n{'='*70}")
    print(f"🔄 Run {run_number}/10 - FloorPlan{fp_number}")
    print(f"{'='*70}")
    
    # Step 1: physical_guard.py 실행
    print(f"\n📝 Step 1: physical_guard.py 실행 중...")
    pg_success = run_physical_guard_subprocess(fp_number, test_file)
    if not pg_success:
        print(f"⚠️  physical_guard.py 실행 실패")
    
    # Step 2: Baseline 실행
    print(f"\n📝 Step 2: Baseline(ProgPrompt).py 실행 중...")
    bl_success = run_baseline_subprocess(fp_number, test_file)
    if not bl_success:
        print(f"⚠️  Baseline 실행 실패")
    
    # Step 3: 결과 파일 찾기
    physical_guard_file, baseline_file = find_result_files(fp_number, folder)
    
    if not physical_guard_file and not baseline_file:
        print(f"⚠️  결과 파일을 찾을 수 없습니다.")
        return {"physical_guard": [], "baseline": []}
    
    # 계획 로드
    physical_guard_plans = {}
    baseline_plans = {}
    
    if physical_guard_file:
        print(f"📖 Physical Guard 계획 로드: {physical_guard_file}")
        physical_guard_plans = parse_json_plan_file(physical_guard_file)
    
    if baseline_file:
        print(f"📖 Baseline 계획 로드: {baseline_file}")
        baseline_plans = parse_json_plan_file(baseline_file)
    
    # 기대 결과 로드
    expected_results = load_expected_results(test_file)
    
    # Executor 초기화
    scene_name = f"FloorPlan{fp_number}"
    physical_guard_exe = None
    baseline_exe = None
    
    if physical_guard_plans:
        physical_guard_exe = AI2ThorExecutor(
            scene=scene_name,
            headless=False,  # 시각화 활성화
            save_video=False
        )
        physical_guard_exe.initialize()
    
    if baseline_plans:
        baseline_exe = AI2ThorExecutor(
            scene=scene_name,
            headless=False,  # 시각화 활성화
            save_video=False
        )
        baseline_exe.initialize()
    
    # 평가 실행
    physical_guard_results = []
    baseline_results = []
    
    for task_def in expected_results:
        task_name = task_def['task']
        task_key = task_name.strip().lower()
        
        # Physical Guard 평가
        if physical_guard_plans and physical_guard_exe:
            matched_key = None
            if task_key in physical_guard_plans:
                matched_key = task_key
            else:
                for plan_key in physical_guard_plans.keys():
                    if task_key in plan_key or plan_key in task_key:
                        matched_key = plan_key
                        break
            
            if matched_key:
                program_code = physical_guard_plans[matched_key]
                result = execute_and_evaluate_task(
                    physical_guard_exe, task_def, program_code, "Physical Guard"
                )
                physical_guard_results.append(result)
        
        # Baseline 평가
        if baseline_plans and baseline_exe:
            matched_key = None
            if task_key in baseline_plans:
                matched_key = task_key
            else:
                for plan_key in baseline_plans.keys():
                    if task_key in plan_key or plan_key in task_key:
                        matched_key = plan_key
                        break
            
            if matched_key:
                program_code = baseline_plans[matched_key]
                result = execute_and_evaluate_task(
                    baseline_exe, task_def, program_code, "Baseline"
                )
                baseline_results.append(result)
    
    # Executor 정리
    if physical_guard_exe:
        physical_guard_exe.cleanup()
    if baseline_exe:
        baseline_exe.cleanup()
    
    return {
        "physical_guard": physical_guard_results,
        "baseline": baseline_results
    }

def calculate_averages(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    모든 실행 결과의 평균을 계산
    
    Args:
        all_results: 각 실행의 결과 리스트
        
    Returns:
        {
            "physical_guard": {
                "task_name": {"avg_gcr": float, "gcr_list": [float, ...], ...},
                ...
            },
            "baseline": {...},
            "scene_avg": {
                "physical_guard": float,
                "baseline": float
            }
        }
    """
    pg_task_results = defaultdict(list)
    bl_task_results = defaultdict(list)
    
    # 각 실행 결과를 task별로 그룹화
    for run_result in all_results:
        for pg_res in run_result.get("physical_guard", []):
            task_name = pg_res.get("task", "Unknown")
            if "gcr" in pg_res:
                pg_task_results[task_name].append(pg_res["gcr"])
        
        for bl_res in run_result.get("baseline", []):
            task_name = bl_res.get("task", "Unknown")
            if "gcr" in bl_res:
                bl_task_results[task_name].append(bl_res["gcr"])
    
    # 평균 계산
    pg_avg = {}
    for task_name, gcr_list in pg_task_results.items():
        pg_avg[task_name] = {
            "avg_gcr": sum(gcr_list) / len(gcr_list),
            "gcr_list": gcr_list,
            "count": len(gcr_list)
        }
    
    bl_avg = {}
    for task_name, gcr_list in bl_task_results.items():
        bl_avg[task_name] = {
            "avg_gcr": sum(gcr_list) / len(gcr_list),
            "gcr_list": gcr_list,
            "count": len(gcr_list)
        }
    
    # Scene 평균 계산
    all_pg_gcr = [r["avg_gcr"] for r in pg_avg.values()]
    all_bl_gcr = [r["avg_gcr"] for r in bl_avg.values()]
    
    scene_avg = {
        "physical_guard": sum(all_pg_gcr) / len(all_pg_gcr) if all_pg_gcr else 0.0,
        "baseline": sum(all_bl_gcr) / len(all_bl_gcr) if all_bl_gcr else 0.0
    }
    
    return {
        "physical_guard": pg_avg,
        "baseline": bl_avg,
        "scene_avg": scene_avg
    }

def save_to_excel(all_scene_results: Dict[str, Dict[str, Any]], output_file: str):
    """
    모든 Scene 결과를 Excel 파일로 저장
    
    Args:
        all_scene_results: {
            "FloorPlan1": {...},
            "FloorPlan216": {...},
            ...
        }
        output_file: 출력 파일 경로
    """
    wb = openpyxl.Workbook()
    
    # 스타일 정의
    header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
    header_font = Font(bold=True, color="FFFFFF")
    center_align = Alignment(horizontal="center", vertical="center")
    
    # 각 Scene별 시트 생성
    for fp_name, scene_data in all_scene_results.items():
        ws = wb.create_sheet(title=fp_name)
        
        # 헤더 작성
        headers = [
            "Task",
            "Physical Guard 평균 GCR (%)",
            "Baseline 평균 GCR (%)",
            "차이 (PG - BL)",
            "Physical Guard GCR (10회)",
            "Baseline GCR (10회)"
        ]
        
        for col_idx, header in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col_idx, value=header)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = center_align
        
        # 데이터 작성
        pg_data = scene_data["physical_guard"]
        bl_data = scene_data["baseline"]
        
        all_tasks = set(pg_data.keys()) | set(bl_data.keys())
        row = 2
        
        for task_name in sorted(all_tasks):
            pg_info = pg_data.get(task_name, {})
            bl_info = bl_data.get(task_name, {})
            
            pg_avg = pg_info.get("avg_gcr", 0.0)
            bl_avg = bl_info.get("avg_gcr", 0.0)
            diff = pg_avg - bl_avg
            
            pg_list = pg_info.get("gcr_list", [])
            bl_list = bl_info.get("gcr_list", [])
            
            # GCR 리스트를 문자열로 변환
            pg_list_str = ", ".join([f"{g:.1f}" for g in pg_list]) if pg_list else "N/A"
            bl_list_str = ", ".join([f"{g:.1f}" for g in bl_list]) if bl_list else "N/A"
            
            ws.cell(row=row, column=1, value=task_name)
            ws.cell(row=row, column=2, value=round(pg_avg, 2))
            ws.cell(row=row, column=3, value=round(bl_avg, 2))
            ws.cell(row=row, column=4, value=round(diff, 2))
            ws.cell(row=row, column=5, value=pg_list_str)
            ws.cell(row=row, column=6, value=bl_list_str)
            
            row += 1
        
        # Scene 평균 행 추가
        row += 1
        scene_avg = scene_data["scene_avg"]
        ws.cell(row=row, column=1, value="Scene 평균")
        ws.cell(row=row, column=2, value=round(scene_avg["physical_guard"], 2))
        ws.cell(row=row, column=3, value=round(scene_avg["baseline"], 2))
        ws.cell(row=row, column=4, value=round(scene_avg["physical_guard"] - scene_avg["baseline"], 2))
        
        # Scene 평균 행 스타일
        for col in range(1, 5):
            cell = ws.cell(row=row, column=col)
            cell.font = Font(bold=True)
            cell.fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
        
        # 열 너비 조정
        ws.column_dimensions['A'].width = 40
        ws.column_dimensions['B'].width = 25
        ws.column_dimensions['C'].width = 25
        ws.column_dimensions['D'].width = 20
        ws.column_dimensions['E'].width = 30
        ws.column_dimensions['F'].width = 30
    
    # 요약 시트 생성
    summary_ws = wb.create_sheet(title="Summary", index=0)
    
    summary_headers = [
        "Scene",
        "Physical Guard 평균 GCR (%)",
        "Baseline 평균 GCR (%)",
        "차이 (PG - BL)",
        "개선율 (%)"
    ]
    
    for col_idx, header in enumerate(summary_headers, 1):
        cell = summary_ws.cell(row=1, column=col_idx, value=header)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = center_align
    
    row = 2
    for fp_name in sorted(all_scene_results.keys()):
        scene_avg = all_scene_results[fp_name]["scene_avg"]
        pg_avg = scene_avg["physical_guard"]
        bl_avg = scene_avg["baseline"]
        diff = pg_avg - bl_avg
        improvement = ((pg_avg - bl_avg) / bl_avg * 100) if bl_avg > 0 else 0.0
        
        summary_ws.cell(row=row, column=1, value=fp_name)
        summary_ws.cell(row=row, column=2, value=round(pg_avg, 2))
        summary_ws.cell(row=row, column=3, value=round(bl_avg, 2))
        summary_ws.cell(row=row, column=4, value=round(diff, 2))
        summary_ws.cell(row=row, column=5, value=round(improvement, 2))
        row += 1
    
    # 전체 평균
    row += 1
    total_pg_avg = sum(r["scene_avg"]["physical_guard"] for r in all_scene_results.values()) / len(all_scene_results)
    total_bl_avg = sum(r["scene_avg"]["baseline"] for r in all_scene_results.values()) / len(all_scene_results)
    total_diff = total_pg_avg - total_bl_avg
    total_improvement = ((total_pg_avg - total_bl_avg) / total_bl_avg * 100) if total_bl_avg > 0 else 0.0
    
    summary_ws.cell(row=row, column=1, value="전체 평균")
    summary_ws.cell(row=row, column=2, value=round(total_pg_avg, 2))
    summary_ws.cell(row=row, column=3, value=round(total_bl_avg, 2))
    summary_ws.cell(row=row, column=4, value=round(total_diff, 2))
    summary_ws.cell(row=row, column=5, value=round(total_improvement, 2))
    
    # 전체 평균 행 스타일
    for col in range(1, 6):
        cell = summary_ws.cell(row=row, column=col)
        cell.font = Font(bold=True, size=12)
        cell.fill = PatternFill(start_color="FFC000", end_color="FFC000", fill_type="solid")
    
    # 열 너비 조정
    for col in range(1, 6):
        summary_ws.column_dimensions[get_column_letter(col)].width = 25
    
    # 기본 시트 제거
    if "Sheet" in wb.sheetnames:
        wb.remove(wb["Sheet"])
    
    # 파일 저장
    wb.save(output_file)
    print(f"\n✅ Excel 파일 저장 완료: {output_file}")

def format_time(seconds: float) -> str:
    """초를 읽기 쉬운 시간 형식으로 변환"""
    if seconds < 60:
        return f"{int(seconds)}초"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}분 {secs}초"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}시간 {minutes}분 {secs}초"

def print_progress(current: int, total: int, start_time: datetime, current_task: str = ""):
    """진행 상황 출력"""
    elapsed = (datetime.now() - start_time).total_seconds()
    progress = (current / total) * 100
    
    if current > 0:
        avg_time_per_task = elapsed / current
        remaining_tasks = total - current
        estimated_remaining = avg_time_per_task * remaining_tasks
        estimated_total = elapsed + estimated_remaining
    else:
        estimated_remaining = 0
        estimated_total = 0
    
    print(f"\n{'='*70}")
    print(f"📊 진행 상황: {current}/{total} ({progress:.1f}%)")
    print(f"⏱️  경과 시간: {format_time(elapsed)}")
    if estimated_remaining > 0:
        print(f"⏳ 예상 남은 시간: {format_time(estimated_remaining)}")
        print(f"🕐 예상 완료 시간: {format_time(estimated_total)}")
    if current_task:
        print(f"🔄 현재 작업: {current_task}")
    print(f"{'='*70}\n")
    
    # 진행 바 표시
    bar_length = 50
    filled = int(bar_length * current / total)
    bar = "█" * filled + "░" * (bar_length - filled)
    print(f"[{bar}] {progress:.1f}%")

def main():
    """메인 함수"""
    print("\n" + "="*70)
    print("🔄 Batch Evaluation Script")
    print("   각 FloorPlan을 10번씩 실행하여 평균 GCR 계산")
    print("="*70)
    
    # 평가할 FloorPlan 목록
    floor_plans = [1, 216, 325, 403]
    num_runs = 10
    
    # 전체 작업 수 계산
    total_tasks = len(floor_plans) * num_runs
    current_task_num = 0
    start_time = datetime.now()
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    all_scene_results = {}
    
    for fp_idx, fp_number in enumerate(floor_plans, 1):
        print(f"\n{'#'*70}")
        print(f"# FloorPlan{fp_number} 평가 시작 ({fp_idx}/{len(floor_plans)})")
        print(f"{'#'*70}")
        
        folder = get_folder_from_fp_number(fp_number)
        test_file = project_root / "data" / "final_test" / f"FloorPlan{fp_number}.json"
        
        if not test_file.exists():
            print(f"⚠️  테스트 파일을 찾을 수 없습니다: {test_file}")
            continue
        
        # 10번 실행
        all_results = []
        for run_num in range(1, num_runs + 1):
            current_task_num += 1
            current_task = f"FloorPlan{fp_number} - Run {run_num}/{num_runs}"
            
            # 진행 상황 출력
            print_progress(current_task_num, total_tasks, start_time, current_task)
            
            try:
                result = run_single_evaluation(
                    fp_number=fp_number,
                    folder=folder,
                    test_file=str(test_file),
                    run_number=run_num
                )
                all_results.append(result)
                print(f"✅ {current_task} 완료")
            except Exception as e:
                print(f"❌ {current_task} 실패: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if not all_results:
            print(f"⚠️  FloorPlan{fp_number}에 대한 유효한 결과가 없습니다.")
            continue
        
        # 평균 계산
        avg_results = calculate_averages(all_results)
        all_scene_results[f"FloorPlan{fp_number}"] = avg_results
        
        print(f"\n✅ FloorPlan{fp_number} 완료 ({fp_idx}/{len(floor_plans)})")
        print(f"   Physical Guard 평균 GCR: {avg_results['scene_avg']['physical_guard']:.2f}%")
        print(f"   Baseline 평균 GCR: {avg_results['scene_avg']['baseline']:.2f}%")
    
    # Excel 파일로 저장
    print(f"\n💾 Excel 파일 저장 중...")
    output_file = project_root / "result_half_test.xlsx"
    save_to_excel(all_scene_results, str(output_file))
    
    # 최종 진행 상황
    total_elapsed = (datetime.now() - start_time).total_seconds()
    print_progress(total_tasks, total_tasks, start_time, "모든 작업 완료")
    
    print("\n" + "="*70)
    print("✅ 모든 평가 완료!")
    print(f"⏱️  총 소요 시간: {format_time(total_elapsed)}")
    print("="*70)

if __name__ == "__main__":
    main()
