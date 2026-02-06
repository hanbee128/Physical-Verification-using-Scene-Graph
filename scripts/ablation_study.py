#!/usr/bin/env python3
# Ablation Study
# (1) Baseline(ProgPrompt)
# (2) Baseline + Physical Guard + Recovery  ← 이 스크립트에서 진행
# (3) Baseline + Physical Guard + Recovery + Task Omission Check/Additional Task Planning
# (1)과 (3)은 batch_evaluation.py 스크립트를 사용하여 평가 실행하였음.
# (2)만 여기서 진행: Baseline 실행 → Baseline 플랜에 Physical Guard + Recovery 적용 → 실행 및 평가 (batch_evaluation과 동일 방식)

import os
import sys
import json
import glob
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict

# scripts 디렉터리를 path에 추가 (physical_guard, evaluate_results import)
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent
# batch_evaluation과 동일 구조: results_ablation/Kitchen/FP1, LivingRoom/FP216 등
ABLATION_RESULTS_BASE = "results_ablation"
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

try:
    import openpyxl
    from openpyxl.styles import Font, Alignment, PatternFill
    from openpyxl.utils import get_column_letter
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl"])
    import openpyxl
    from openpyxl.styles import Font, Alignment, PatternFill
    from openpyxl.utils import get_column_letter


def get_folder_from_fp_number(fp_number: int) -> str:
    if fp_number in [1, 2]:
        return "Kitchen"
    elif fp_number in [216, 224]:
        return "LivingRoom"
    elif fp_number in [325, 326]:
        return "BedRoom"
    elif fp_number in [403, 425]:
        return "BathRoom"
    return "Kitchen"


def load_tasks_from_fp(project_root: Path, fp_number: int) -> List[Dict[str, Any]]:
    path = project_root / "data" / "final_test" / f"FloorPlan{fp_number}.json"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "task" in data:
        return [data]
    return []


def run_baseline_subprocess(fp_number: int, task_file: str, folder: str) -> bool:
    """Baseline(ProgPrompt).py를 subprocess로 실행. batch_evaluation과 동일."""
    baseline_script = _script_dir / "Baseline(ProgPrompt).py"
    if not baseline_script.exists():
        print(f"⚠️  Baseline 스크립트를 찾을 수 없습니다: {baseline_script}")
        return False
    out_dir = _project_root / ABLATION_RESULTS_BASE / folder / f"FP{fp_number}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(baseline_script),
        "--scene-number", str(fp_number),
        "--output-dir", str(out_dir),
        "--task-file", task_file,
    ]
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            cwd=_project_root,
        )
        for line in process.stdout:
            print(line.rstrip())
        process.wait()
        return process.returncode == 0
    except Exception as e:
        print(f"❌ Baseline 실행 실패: {e}")
        return False


def find_latest_baseline_json(fp_number: int, folder: str) -> str:
    """results_ablation/{folder}/FP{fp_number}/ 내 최신 baseline_result_*.json 경로 반환."""
    base = _project_root / ABLATION_RESULTS_BASE / folder / f"FP{fp_number}"
    if not base.exists():
        return ""
    files = glob.glob(str(base / "baseline_result_*.json"))
    if not files:
        return ""
    return max(files, key=os.path.getmtime)


def run_ablation_baseline_pg_recovery(
    fp_number: int,
    folder: str,
    test_file: str,
    baseline_json_path: str,
) -> str:
    """
    Baseline 플랜을 로드한 뒤, 각 task에 대해 Physical Guard + Recovery만 적용.
    결과를 ablation_baseline_pg_recovery_result.json으로 저장하고 해당 경로 반환.
    """
    from evaluate_results import load_expected_results, parse_json_plan_file
    from physical_guard import load_scene_graph, generate_final_plan_with_physical_verification

    # Baseline 플랜 로드 (원본 키 유지용 raw + normalize용)
    with open(baseline_json_path, "r", encoding="utf-8") as f:
        baseline_raw = json.load(f)
    baseline_plans_norm = parse_json_plan_file(baseline_json_path)

    expected_results = load_expected_results(test_file)
    scene_graph_path = _script_dir / f"scene_graph_structured_FloorPlan{fp_number}.json"
    if not scene_graph_path.exists():
        print(f"⚠️  Scene Graph 없음: {scene_graph_path}")
        return ""

    out_dir = _project_root / ABLATION_RESULTS_BASE / folder / f"FP{fp_number}"
    out_dir.mkdir(parents=True, exist_ok=True)
    updated_sg_path = out_dir / "updated_scene_graph_ablation.json"
    shutil.copy2(str(scene_graph_path), str(updated_sg_path))
    scene_graph = load_scene_graph(str(updated_sg_path))

    controller = None
    try:
        from ai2thor.controller import Controller
        scene_name = f"FloorPlan{fp_number}_physics"
        controller = Controller(
            agentMode="arm",
            scene=scene_name,
            gridSize=0.25,
            snapToGrid=False,
            rotateStepDegrees=90,
            visibilityDistance=1.5,
            renderInstanceSegmentation=False,
            renderDepthImage=False,
            renderSemanticSegmentation=False,
            width=300,
            height=300,
            fieldOfView=90,
        )
    except Exception as e:
        print(f"  ⚠️  Controller 초기화 실패 (NavMesh 제한): {e}")

    def _norm(s: str) -> str:
        return (s or "").strip().lower().replace(" ", "")

    ablation_plans = {}
    for task_def in expected_results:
        task_name = task_def.get("task", "")
        task_key = task_name.strip().lower()
        task_norm = _norm(task_name)

        # task마다 scene graph 초기 상태로 리셋 (독립 실행)
        shutil.copy2(str(scene_graph_path), str(updated_sg_path))
        scene_graph = load_scene_graph(str(updated_sg_path))

        baseline_code = baseline_raw.get(task_name) or baseline_raw.get(task_key)
        if not baseline_code and baseline_plans_norm:
            baseline_code = baseline_plans_norm.get(task_key)
            if not baseline_code:
                for k, v in baseline_plans_norm.items():
                    if _norm(k) == task_norm:
                        baseline_code = v
                        break
        if not baseline_code:
            print(f"  ⚠️  Baseline plan 없음: '{task_name}'")
            continue

        try:
            final_program, _ = generate_final_plan_with_physical_verification(
                task=task_name,
                initial_program=baseline_code,
                scene_graph=scene_graph,
                controller=controller,
                client=None,
                model="llama3",
                scene_graph_path=str(updated_sg_path),
            )
            if final_program:
                ablation_plans[task_name] = final_program
        except Exception as e:
            print(f"  ⚠️  Physical verification 실패 '{task_name}': {e}")

    if controller is not None:
        try:
            controller.stop()
        except Exception:
            pass

    out_path = out_dir / "ablation_baseline_pg_recovery_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ablation_plans, f, indent=2, ensure_ascii=False)
    return str(out_path)


def run_evaluate_ablation_vs_baseline(
    fp_number: int,
    folder: str,
    test_file: str,
    ablation_json_path: str,
    baseline_json_path: str,
) -> Dict[str, Any]:
    """evaluate_results 로직을 ablation_study 내에서 직접 수행 (ablation=PG, baseline=BL). subprocess/폴더 검사 없이 동작."""
    from evaluate_results import load_expected_results, parse_json_plan_file, execute_and_evaluate_task
    from ai2thor_connector_ithor import AI2ThorExecutor

    ablation_plans = parse_json_plan_file(ablation_json_path)
    baseline_plans = parse_json_plan_file(baseline_json_path)
    expected_results = load_expected_results(test_file)
    scene_name = f"FloorPlan{fp_number}"

    def _norm(s: str) -> str:
        return (s or "").lower().strip().replace(" ", "")

    physical_guard_exe = None
    baseline_exe = None
    if ablation_plans:
        physical_guard_exe = AI2ThorExecutor(scene=scene_name, headless=False, save_video=False)
        physical_guard_exe.initialize()
    if baseline_plans:
        baseline_exe = AI2ThorExecutor(scene=scene_name, headless=False, save_video=False)
        baseline_exe.initialize()

    physical_guard_results = []
    baseline_results = []
    for task_def in expected_results:
        task_name = task_def["task"]
        task_key = task_name.strip().lower()

        if ablation_plans and physical_guard_exe:
            matched_key = None
            if task_key in ablation_plans:
                matched_key = task_key
            else:
                task_norm = _norm(task_name)
                for plan_key in ablation_plans.keys():
                    if _norm(plan_key) == task_norm:
                        matched_key = plan_key
                        break
                if not matched_key:
                    for plan_key in ablation_plans.keys():
                        if task_key in plan_key or plan_key in task_key:
                            matched_key = plan_key
                            break
            if matched_key:
                result = execute_and_evaluate_task(
                    physical_guard_exe, task_def, ablation_plans[matched_key], "Physical Guard"
                )
                physical_guard_results.append(result)
            else:
                physical_guard_results.append({
                    "task": task_name, "method": "Physical Guard", "status": "SKIPPED", "reason": "No Plan"
                })

        if baseline_plans and baseline_exe:
            matched_key = None
            if task_key in baseline_plans:
                matched_key = task_key
            else:
                task_norm = _norm(task_name)
                for plan_key in baseline_plans.keys():
                    if _norm(plan_key) == task_norm:
                        matched_key = plan_key
                        break
                if not matched_key:
                    for plan_key in baseline_plans.keys():
                        if task_key in plan_key or plan_key in task_key:
                            matched_key = plan_key
                            break
            if matched_key:
                result = execute_and_evaluate_task(
                    baseline_exe, task_def, baseline_plans[matched_key], "Baseline"
                )
                baseline_results.append(result)
            else:
                baseline_results.append({
                    "task": task_name, "method": "Baseline", "status": "SKIPPED", "reason": "No Plan"
                })

    if physical_guard_exe:
        try:
            physical_guard_exe.close()
        except Exception:
            pass
    if baseline_exe:
        try:
            baseline_exe.close()
        except Exception:
            pass

    result_dict = {"physical_guard": physical_guard_results, "baseline": baseline_results}
    output_json = _project_root / ABLATION_RESULTS_BASE / folder / f"FP{fp_number}" / "_ablation_eval_result.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)
    return result_dict


def calculate_averages(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """batch_evaluation과 동일: physical_guard=ablation, baseline=baseline."""
    pg_task_results = defaultdict(list)
    bl_task_results = defaultdict(list)
    pg_task_exec = defaultdict(list)
    bl_task_exec = defaultdict(list)
    for run_result in all_results:
        for r in run_result.get("physical_guard", []):
            task_name = r.get("task", "Unknown")
            if "gcr" in r:
                pg_task_results[task_name].append(r["gcr"])
            if "exec" in r:
                pg_task_exec[task_name].append(r["exec"])
        for r in run_result.get("baseline", []):
            task_name = r.get("task", "Unknown")
            if "gcr" in r:
                bl_task_results[task_name].append(r["gcr"])
            if "exec" in r:
                bl_task_exec[task_name].append(r["exec"])
    pg_avg = {}
    for task_name, gcr_list in pg_task_results.items():
        exec_list = pg_task_exec.get(task_name, [])
        pg_avg[task_name] = {
            "avg_gcr": sum(gcr_list) / len(gcr_list),
            "gcr_list": gcr_list,
            "avg_exec": sum(exec_list) / len(exec_list) if exec_list else 0.0,
            "exec_list": exec_list,
            "count": len(gcr_list),
        }
    bl_avg = {}
    for task_name, gcr_list in bl_task_results.items():
        exec_list = bl_task_exec.get(task_name, [])
        bl_avg[task_name] = {
            "avg_gcr": sum(gcr_list) / len(gcr_list),
            "gcr_list": gcr_list,
            "avg_exec": sum(exec_list) / len(exec_list) if exec_list else 0.0,
            "exec_list": exec_list,
            "count": len(gcr_list),
        }
    all_pg_gcr = [r["avg_gcr"] for r in pg_avg.values()]
    all_bl_gcr = [r["avg_gcr"] for r in bl_avg.values()]
    all_pg_exec = [r["avg_exec"] for r in pg_avg.values()]
    all_bl_exec = [r["avg_exec"] for r in bl_avg.values()]
    scene_avg = {
        "physical_guard": sum(all_pg_gcr) / len(all_pg_gcr) if all_pg_gcr else 0.0,
        "baseline": sum(all_bl_gcr) / len(all_bl_gcr) if all_bl_gcr else 0.0,
        "physical_guard_exec": sum(all_pg_exec) / len(all_pg_exec) if all_pg_exec else 0.0,
        "baseline_exec": sum(all_bl_exec) / len(all_bl_exec) if all_bl_exec else 0.0,
    }
    return {"physical_guard": pg_avg, "baseline": bl_avg, "scene_avg": scene_avg}


def save_to_excel(
    all_scene_results: Dict[str, Dict[str, Any]],
    all_run_results: Dict[str, List[Dict[str, Any]]],
    output_file: str,
    fp_number: int | None = None,
    task_index: int | None = None,
) -> None:
    """batch_evaluation과 동일 구조. Physical Guard 컬럼 = Ablation (Baseline+PG+Recovery).
    fp_number, task_index가 모두 주어지면 시트 제목에 FP번호_Task번호 포함 (단일 task 실행 시)."""
    wb = openpyxl.Workbook()
    header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
    header_font = Font(bold=True, color="FFFFFF")
    center_align = Alignment(horizontal="center", vertical="center")

    # 단일 task 실행 시 시트 제목에 FP번호_Task번호 포함, 전체(0) 실행 시 FP만
    def _sheet_title_run(fp_name: str, run_num: int) -> str:
        if fp_number is not None and task_index is not None:
            return f"FP{fp_number}_Task{task_index}_R{run_num}"[:31]
        return f"{fp_name}_R{run_num}"[:31]

    def _sheet_title_avg(fp_name: str) -> str:
        if fp_number is not None and task_index is not None:
            return f"FP{fp_number}_Task{task_index}_Avg"[:31]
        return f"{fp_name}_Avg"[:31]

    for fp_name, run_list in all_run_results.items():
        for run_data in run_list:
            run_num = run_data["run_number"]
            result = run_data["result"]
            sheet_name = _sheet_title_run(fp_name, run_num)
            ws = wb.create_sheet(title=sheet_name)
            headers = [
                "Task",
                "Ablation(BL+PG+Rec) GCR (%)", "Baseline GCR (%)", "차이",
                "Ablation Exec (%)", "Baseline Exec (%)", "차이 Exec",
            ]
            for col_idx, header in enumerate(headers, 1):
                c = ws.cell(row=1, column=col_idx, value=header)
                c.fill = header_fill
                c.font = header_font
                c.alignment = center_align
            pg_results = {r["task"]: r for r in result.get("physical_guard", [])}
            bl_results = {r["task"]: r for r in result.get("baseline", [])}
            all_tasks = set(pg_results.keys()) | set(bl_results.keys())
            row = 2
            for task_name in sorted(all_tasks):
                pg_res = pg_results.get(task_name, {})
                bl_res = bl_results.get(task_name, {})
                pg_gcr = pg_res.get("gcr", 0.0) if pg_res else 0.0
                bl_gcr = bl_res.get("gcr", 0.0) if bl_res else 0.0
                pg_exec = pg_res.get("exec", 0.0) if pg_res else 0.0
                bl_exec = bl_res.get("exec", 0.0) if bl_res else 0.0
                ws.cell(row=row, column=1, value=task_name)
                ws.cell(row=row, column=2, value=round(pg_gcr, 2))
                ws.cell(row=row, column=3, value=round(bl_gcr, 2))
                ws.cell(row=row, column=4, value=round(pg_gcr - bl_gcr, 2))
                ws.cell(row=row, column=5, value=round(pg_exec, 2))
                ws.cell(row=row, column=6, value=round(bl_exec, 2))
                ws.cell(row=row, column=7, value=round(pg_exec - bl_exec, 2))
                row += 1
            ws.column_dimensions["A"].width = 40
            for col in ["B", "C", "D", "E", "F", "G"]:
                ws.column_dimensions[col].width = 18

    for fp_name, scene_data in all_scene_results.items():
        ws = wb.create_sheet(title=_sheet_title_avg(fp_name))
        headers = [
            "Task", "Ablation 평균 GCR (%)", "Baseline 평균 GCR (%)", "차이",
            "Ablation GCR (10회)", "Baseline GCR (10회)",
            "Ablation 평균 Exec (%)", "Baseline 평균 Exec (%)", "차이 Exec",
            "Ablation Exec (10회)", "Baseline Exec (10회)",
        ]
        for col_idx, header in enumerate(headers, 1):
            c = ws.cell(row=1, column=col_idx, value=header)
            c.fill = header_fill
            c.font = header_font
            c.alignment = center_align
        row = 2
        pg_avg = scene_data.get("physical_guard", {})
        bl_avg = scene_data.get("baseline", {})
        scene_avg = scene_data.get("scene_avg", {})
        for task_name in sorted(set(pg_avg.keys()) | set(bl_avg.keys())):
            pg_info = pg_avg.get(task_name, {})
            bl_info = bl_avg.get(task_name, {})
            pg_gcr = pg_info.get("avg_gcr", 0.0)
            bl_gcr = bl_info.get("avg_gcr", 0.0)
            pg_exec = pg_info.get("avg_exec", 0.0)
            bl_exec = bl_info.get("avg_exec", 0.0)
            ws.cell(row=row, column=1, value=task_name)
            ws.cell(row=row, column=2, value=round(pg_gcr, 2))
            ws.cell(row=row, column=3, value=round(bl_gcr, 2))
            ws.cell(row=row, column=4, value=round(pg_gcr - bl_gcr, 2))
            ws.cell(row=row, column=5, value=", ".join([f"{g:.1f}" for g in pg_info.get("gcr_list", [])]))
            ws.cell(row=row, column=6, value=", ".join([f"{g:.1f}" for g in bl_info.get("gcr_list", [])]))
            ws.cell(row=row, column=7, value=round(pg_exec, 2))
            ws.cell(row=row, column=8, value=round(bl_exec, 2))
            ws.cell(row=row, column=9, value=round(pg_exec - bl_exec, 2))
            ws.cell(row=row, column=10, value=", ".join([f"{e:.1f}" for e in pg_info.get("exec_list", [])]))
            ws.cell(row=row, column=11, value=", ".join([f"{e:.1f}" for e in bl_info.get("exec_list", [])]))
            row += 1
        row += 1
        ws.cell(row=row, column=1, value="Scene 평균")
        ws.cell(row=row, column=2, value=round(scene_avg.get("physical_guard", 0), 2))
        ws.cell(row=row, column=3, value=round(scene_avg.get("baseline", 0), 2))
        ws.cell(row=row, column=4, value=round(scene_avg.get("physical_guard", 0) - scene_avg.get("baseline", 0), 2))
        ws.cell(row=row, column=7, value=round(scene_avg.get("physical_guard_exec", 0), 2))
        ws.cell(row=row, column=8, value=round(scene_avg.get("baseline_exec", 0), 2))
        ws.cell(row=row, column=9, value=round(scene_avg.get("physical_guard_exec", 0) - scene_avg.get("baseline_exec", 0), 2))
        for col in range(1, 12):
            cell = ws.cell(row=row, column=col)
            cell.font = Font(bold=True)
            cell.fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
        ws.column_dimensions["A"].width = 40
        for c in range(2, 12):
            ws.column_dimensions[get_column_letter(c)].width = 22

    summary_ws = wb.create_sheet(title="Summary", index=0)
    summary_headers = [
        "Scene", "Ablation(BL+PG+Rec) 평균 GCR (%)", "Baseline 평균 GCR (%)", "차이", "개선율 (%)",
        "Ablation 평균 Exec (%)", "Baseline 평균 Exec (%)", "차이 Exec", "개선율 Exec (%)",
    ]
    for col_idx, header in enumerate(summary_headers, 1):
        c = summary_ws.cell(row=1, column=col_idx, value=header)
        c.fill = header_fill
        c.font = header_font
        c.alignment = center_align
    row = 2
    n = len(all_scene_results)
    for fp_name in sorted(all_scene_results.keys()):
        scene_avg = all_scene_results[fp_name]["scene_avg"]
        pg_gcr = scene_avg.get("physical_guard", 0.0)
        bl_gcr = scene_avg.get("baseline", 0.0)
        pg_exec = scene_avg.get("physical_guard_exec", 0.0)
        bl_exec = scene_avg.get("baseline_exec", 0.0)
        diff_gcr = pg_gcr - bl_gcr
        diff_exec = pg_exec - bl_exec
        imp_gcr = (diff_gcr / bl_gcr * 100) if bl_gcr > 0 else 0.0
        imp_exec = (diff_exec / bl_exec * 100) if bl_exec > 0 else 0.0
        summary_ws.cell(row=row, column=1, value=fp_name)
        summary_ws.cell(row=row, column=2, value=round(pg_gcr, 2))
        summary_ws.cell(row=row, column=3, value=round(bl_gcr, 2))
        summary_ws.cell(row=row, column=4, value=round(diff_gcr, 2))
        summary_ws.cell(row=row, column=5, value=round(imp_gcr, 2))
        summary_ws.cell(row=row, column=6, value=round(pg_exec, 2))
        summary_ws.cell(row=row, column=7, value=round(bl_exec, 2))
        summary_ws.cell(row=row, column=8, value=round(diff_exec, 2))
        summary_ws.cell(row=row, column=9, value=round(imp_exec, 2))
        row += 1
    row += 1
    if n > 0:
        total_pg_gcr = sum(r["scene_avg"]["physical_guard"] for r in all_scene_results.values()) / n
        total_bl_gcr = sum(r["scene_avg"]["baseline"] for r in all_scene_results.values()) / n
        total_pg_exec = sum(r["scene_avg"].get("physical_guard_exec", 0) for r in all_scene_results.values()) / n
        total_bl_exec = sum(r["scene_avg"].get("baseline_exec", 0) for r in all_scene_results.values()) / n
    else:
        total_pg_gcr = total_bl_gcr = total_pg_exec = total_bl_exec = 0.0
    summary_ws.cell(row=row, column=1, value="전체 평균")
    summary_ws.cell(row=row, column=2, value=round(total_pg_gcr, 2))
    summary_ws.cell(row=row, column=3, value=round(total_bl_gcr, 2))
    summary_ws.cell(row=row, column=4, value=round(total_pg_gcr - total_bl_gcr, 2))
    summary_ws.cell(row=row, column=5, value=round(((total_pg_gcr - total_bl_gcr) / total_bl_gcr * 100) if total_bl_gcr > 0 else 0, 2))
    summary_ws.cell(row=row, column=6, value=round(total_pg_exec, 2))
    summary_ws.cell(row=row, column=7, value=round(total_bl_exec, 2))
    summary_ws.cell(row=row, column=8, value=round(total_pg_exec - total_bl_exec, 2))
    summary_ws.cell(row=row, column=9, value=round(((total_pg_exec - total_bl_exec) / total_bl_exec * 100) if total_bl_exec > 0 else 0, 2))
    for col in range(1, 10):
        cell = summary_ws.cell(row=row, column=col)
        cell.font = Font(bold=True, size=12)
        cell.fill = PatternFill(start_color="FFC000", end_color="FFC000", fill_type="solid")
    for col in range(1, 10):
        summary_ws.column_dimensions[get_column_letter(col)].width = 25
    if "Sheet" in wb.sheetnames:
        wb.remove(wb["Sheet"])
    wb.save(output_file)
    print(f"\n✅ Excel 저장 완료: {output_file}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ablation (2): Baseline + Physical Guard + Recovery 실행 및 평가")
    parser.add_argument("--fp-number", type=int, default=None, help="FloorPlan 번호 (미지정 시 1, 2, 216, 224, 325, 326, 403, 425)")
    parser.add_argument("--num-runs", type=int, default=10, help="실행 횟수 (기본 10)")
    args = parser.parse_args()

    floor_plans = [args.fp_number] if args.fp_number else [1, 2, 216, 224, 325, 326, 403, 425]
    num_runs = args.num_runs

    # --fp-number 지정 시: 해당 FP의 Task 목록을 보여주고 번호로 선택 (0=전체, 1~N=단일 Task)
    selected_task_index = None
    single_task_path = None
    if args.fp_number:
        task_list = load_tasks_from_fp(_project_root, args.fp_number)
        if not task_list:
            print(f"❌ FloorPlan{args.fp_number}의 Task 파일을 읽을 수 없거나 비어 있습니다.")
            sys.exit(1)
        n_tasks = len(task_list)
        print(f"\n📋 FloorPlan{args.fp_number} 의 Task 목록 (총 {n_tasks}개):")
        for i, t in enumerate(task_list, 1):
            task_str = t.get("task", "(task 없음)")
            print(f"   {i}. {task_str}")
        while True:
            try:
                sel = input(f"\n실행할 Task 번호를 선택하세요 (0=해당 FP 전체 Task, 1-{n_tasks}=단일 Task): ").strip()
                task_index = int(sel)
                if task_index == 0:
                    selected_task_index = None
                    single_task_path = None
                    print(f"   선택: 0 — 해당 FP 전체 Task {n_tasks}개 {num_runs}번 실행")
                    break
                if 1 <= task_index <= n_tasks:
                    selected_task_def = task_list[task_index - 1]
                    single_task_path = _project_root / "data" / "final_test" / f"_single_task_fp{args.fp_number}_task{task_index}.json"
                    single_task_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(single_task_path, "w", encoding="utf-8") as f:
                        json.dump([selected_task_def], f, indent=2, ensure_ascii=False)
                    selected_task_index = task_index
                    print(f"   선택: Task {task_index} — {num_runs}번 실행 후 결과 저장")
                    break
            except ValueError:
                pass
            print(f"   잘못된 입력입니다. 0 또는 1~{n_tasks} 사이 숫자를 입력하세요.")

    print("\n" + "=" * 70)
    print("🔬 Ablation Study (2): Baseline + Physical Guard + Recovery")
    print(f"   FloorPlans: {floor_plans}, Runs: {num_runs}")
    print("=" * 70)

    all_scene_results = {}
    all_run_results = defaultdict(list)
    start_time = datetime.now()

    try:
        for fp_number in floor_plans:
            folder = get_folder_from_fp_number(fp_number)
            # 단일 Task 선택 시 해당 task만 담긴 JSON 사용, 아니면 전체 FloorPlan JSON 사용
            if args.fp_number and selected_task_index is not None and single_task_path is not None:
                test_file = str(single_task_path)
            else:
                test_file = str(_project_root / "data" / "final_test" / f"FloorPlan{fp_number}.json")
            if not Path(test_file).exists():
                print(f"⚠️  테스트 파일 없음: {test_file}")
                continue
            fp_name = f"FloorPlan{fp_number}"
            all_run_results[fp_name] = []

            for run_num in range(1, num_runs + 1):
                print(f"\n{'='*70}")
                print(f"📌 Run {run_num}/{num_runs} - {fp_name}")
                print("=" * 70)

                # 1) Baseline 실행
                print("\n📝 Step 1: Baseline 실행 중...")
                if not run_baseline_subprocess(fp_number, test_file, folder):
                    print("⚠️  Baseline 실패, 해당 run 스킵")
                    continue
                baseline_json = find_latest_baseline_json(fp_number, folder)
                if not baseline_json:
                    print("⚠️  Baseline JSON을 찾을 수 없음")
                    continue

                # 2) Baseline + Physical Guard + Recovery 적용
                print("\n📝 Step 2: Baseline + Physical Guard + Recovery 적용 중...")
                ablation_json = run_ablation_baseline_pg_recovery(
                    fp_number=fp_number,
                    folder=folder,
                    test_file=test_file,
                    baseline_json_path=baseline_json,
                )
                if not ablation_json:
                    print("⚠️  Ablation 결과 저장 실패")
                    continue

                # 3) 실행 및 평가 (batch_evaluation과 동일)
                print("\n📝 Step 3: 실행 및 평가 (evaluate_results)...")
                result = run_evaluate_ablation_vs_baseline(
                    fp_number=fp_number,
                    folder=folder,
                    test_file=test_file,
                    ablation_json_path=ablation_json,
                    baseline_json_path=baseline_json,
                )
                all_run_results[fp_name].append({"run_number": run_num, "result": result})
                print(f"✅ Run {run_num} 완료")

            if all_run_results[fp_name]:
                avg_results = calculate_averages([r["result"] for r in all_run_results[fp_name]])
                all_scene_results[fp_name] = avg_results
                print(f"\n✅ {fp_name} 완료 (평균 GCR Ablation: {avg_results['scene_avg']['physical_guard']:.2f}%, Baseline: {avg_results['scene_avg']['baseline']:.2f}%)")

        if all_scene_results:
            out_dir = _project_root / ABLATION_RESULTS_BASE
            out_dir.mkdir(parents=True, exist_ok=True)
            # batch_evaluation과 동일: 단일 task 선택 시 FP번호_task번호, 전체(0) 선택 시 FP번호만
            if args.fp_number and selected_task_index is not None:
                output_file = out_dir / f"FP{args.fp_number}_task{selected_task_index}_ablation_result.xlsx"
            elif args.fp_number:
                output_file = out_dir / f"FP{args.fp_number}_ablation_result.xlsx"
            else:
                output_file = out_dir / "ablation_baseline_pg_recovery_result.xlsx"
            save_to_excel(
                all_scene_results,
                dict(all_run_results),
                str(output_file),
                fp_number=args.fp_number if args.fp_number else None,
                task_index=selected_task_index,
            )
            print(f"   저장된 파일: {output_file}")

        elapsed = (datetime.now() - start_time).total_seconds()
        print("\n" + "=" * 70)
        print("✅ Ablation Study (2): Baseline + Physical Guard + Recovery 완료")
        print(f"⏱️  소요 시간: {int(elapsed // 60)}분 {int(elapsed % 60)}초")
        print("=" * 70)

    finally:
        # 단일 Task용 임시 JSON 삭제
        if single_task_path is not None and Path(single_task_path).exists():
            try:
                Path(single_task_path).unlink()
            except Exception:
                pass


if __name__ == "__main__":
    main()
