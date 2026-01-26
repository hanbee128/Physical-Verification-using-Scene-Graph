#!/usr/bin/env python3
"""
physical_guard.py와 evaluate_results.py를 순차적으로 실행하는 통합 스크립트

사용자가 Scene 번호를 입력하면:
1. physical_guard.py를 해당 Scene 번호로 실행
2. physical_guard.py 실행 완료 후 evaluate_results.py를 해당 FP 번호로 실행
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

def get_folder_from_fp_number(fp_number: int) -> str:
    """
    FloorPlan 번호에 따라 폴더 이름을 반환합니다.
    
    Args:
        fp_number: FloorPlan 번호
        
    Returns:
        폴더 이름 (Kitchen, LivingRoom, BedRoom, BathRoom)
    """
    if fp_number in [1, 2]:
        return "Kitchen"
    elif fp_number in [216, 224]:
        return "LivingRoom"
    elif fp_number in [325, 326]:
        return "BedRoom"
    elif fp_number in [403, 425]:
        return "BathRoom"
    else:
        # 기본값으로 Kitchen 사용
        print(f"⚠️  알려지지 않은 FloorPlan 번호: {fp_number}, 기본값 Kitchen 사용")
        return "Kitchen"

def run_physical_guard(fp_number: int, task_file: str = None, tasks: list = None, log_buffer: list = None):
    """
    physical_guard.py를 실행합니다.
    
    Args:
        fp_number: FloorPlan 번호
        task_file: 작업 파일 경로 (선택사항)
        tasks: 작업 목록 (선택사항)
        log_buffer: 로그를 저장할 리스트 (선택사항)
        
    Returns:
        (실행 성공 여부, 출력 로그)
    """
    header = f"\n{'='*60}\n🚀 Step 1: physical_guard.py 실행 (FloorPlan {fp_number})\n{'='*60}\n"
    print(header)
    if log_buffer is not None:
        log_buffer.append(header)
    
    # 스크립트 경로
    script_dir = Path(__file__).parent
    physical_guard_script = script_dir / "physical_guard.py"
    
    if not physical_guard_script.exists():
        error_msg = f"❌ 오류: {physical_guard_script} 파일을 찾을 수 없습니다."
        print(error_msg)
        if log_buffer is not None:
            log_buffer.append(error_msg)
        return False, ""
    
    # 명령어 구성
    cmd = [sys.executable, str(physical_guard_script), "--scene-number", str(fp_number)]
    
    # task_file이 있으면 추가
    if task_file:
        cmd.extend(["--task-file", task_file])
    
    # tasks가 있으면 추가
    if tasks:
        cmd.extend(["--tasks"] + tasks)
    
    try:
        # physical_guard.py 실행 (실시간 출력 및 로그 캡처)
        output_lines = []
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            cwd=script_dir.parent
        )
        
        # 실시간으로 출력하면서 로그 저장
        for line in process.stdout:
            line = line.rstrip()
            print(line)
            output_lines.append(line)
            if log_buffer is not None:
                log_buffer.append(line)
        
        process.wait()
        output = "\n".join(output_lines)
        
        if process.returncode == 0:
            success_msg = "\n✅ physical_guard.py 실행 완료!"
            print(success_msg)
            if log_buffer is not None:
                log_buffer.append(success_msg)
            return True, output
        else:
            error_msg = f"\n❌ physical_guard.py 실행 실패 (반환 코드: {process.returncode})"
            print(error_msg)
            if log_buffer is not None:
                log_buffer.append(error_msg)
            return False, output
            
    except KeyboardInterrupt:
        error_msg = "\n⚠️  physical_guard.py 실행이 사용자에 의해 중단되었습니다."
        print(error_msg)
        if log_buffer is not None:
            log_buffer.append(error_msg)
        return False, ""
    except Exception as e:
        error_msg = f"\n❌ physical_guard.py 실행 중 오류 발생: {e}"
        print(error_msg)
        if log_buffer is not None:
            log_buffer.append(error_msg)
        return False, ""

def run_evaluate_results(fp_number: int, folder: str = None, log_buffer: list = None):
    """
    evaluate_results.py를 실행합니다.
    
    Args:
        fp_number: FloorPlan 번호
        folder: 폴더 이름 (선택사항, 없으면 자동 결정)
        log_buffer: 로그를 저장할 리스트 (선택사항)
        
    Returns:
        (실행 성공 여부, 출력 로그)
    """
    header = f"\n{'='*60}\n📊 Step 2: evaluate_results.py 실행 (FloorPlan {fp_number})\n{'='*60}\n"
    print(header)
    if log_buffer is not None:
        log_buffer.append(header)
    
    # 스크립트 경로
    script_dir = Path(__file__).parent
    evaluate_script = script_dir / "evaluate_results.py"
    
    if not evaluate_script.exists():
        error_msg = f"❌ 오류: {evaluate_script} 파일을 찾을 수 없습니다."
        print(error_msg)
        if log_buffer is not None:
            log_buffer.append(error_msg)
        return False, ""
    
    # 폴더가 지정되지 않았으면 자동 결정
    if folder is None:
        folder = get_folder_from_fp_number(fp_number)
    
    # 명령어 구성
    cmd = [
        sys.executable,
        str(evaluate_script),
        "--fp-number", str(fp_number),
        "--folder", folder
    ]
    
    try:
        # evaluate_results.py 실행 (실시간 출력 및 로그 캡처)
        output_lines = []
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            cwd=script_dir.parent
        )
        
        # 실시간으로 출력하면서 로그 저장
        for line in process.stdout:
            line = line.rstrip()
            print(line)
            output_lines.append(line)
            if log_buffer is not None:
                log_buffer.append(line)
        
        process.wait()
        output = "\n".join(output_lines)
        
        if process.returncode == 0:
            success_msg = "\n✅ evaluate_results.py 실행 완료!"
            print(success_msg)
            if log_buffer is not None:
                log_buffer.append(success_msg)
            return True, output
        else:
            error_msg = f"\n❌ evaluate_results.py 실행 실패 (반환 코드: {process.returncode})"
            print(error_msg)
            if log_buffer is not None:
                log_buffer.append(error_msg)
            return False, output
            
    except KeyboardInterrupt:
        error_msg = "\n⚠️  evaluate_results.py 실행이 사용자에 의해 중단되었습니다."
        print(error_msg)
        if log_buffer is not None:
            log_buffer.append(error_msg)
        return False, ""
    except Exception as e:
        error_msg = f"\n❌ evaluate_results.py 실행 중 오류 발생: {e}"
        print(error_msg)
        if log_buffer is not None:
            log_buffer.append(error_msg)
        return False, ""

def main():
    """메인 함수"""
    # 로그 버퍼 초기화
    log_buffer = []
    
    # 시작 시간 기록
    start_time = datetime.now()
    start_header = f"\n{'='*60}\n🔧 Unified Execution Script\n   physical_guard.py와 evaluate_results.py를 순차적으로 실행합니다\n{'='*60}\n"
    start_header += f"실행 시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    print(start_header)
    log_buffer.append(start_header)
    
    # Scene 번호 입력 받기
    try:
        fp_number_input = input("\n📋 FloorPlan 번호를 입력하세요 (예: 1, 2, 216, 224, 325, 326, 403, 425): ").strip()
        log_buffer.append(f"\n📋 FloorPlan 번호 입력: {fp_number_input}\n")
        if not fp_number_input:
            error_msg = "❌ FloorPlan 번호가 입력되지 않았습니다."
            print(error_msg)
            log_buffer.append(error_msg)
            save_log_to_file(log_buffer, start_time)
            sys.exit(1)
        
        fp_number = int(fp_number_input)
    except ValueError:
        error_msg = "❌ 잘못된 입력입니다. 숫자를 입력해주세요."
        print(error_msg)
        log_buffer.append(error_msg)
        save_log_to_file(log_buffer, start_time)
        sys.exit(1)
    except (KeyboardInterrupt, EOFError):
        error_msg = "\n❌ 입력이 취소되었습니다."
        print(error_msg)
        log_buffer.append(error_msg)
        save_log_to_file(log_buffer, start_time)
        sys.exit(1)
    
    # 작업 파일 경로 확인 (자동으로 찾기)
    task_file = None
    task_file_path = Path(f"data/final_test/FloorPlan{fp_number}.json")
    if task_file_path.exists():
        task_file = str(task_file_path)
        msg = f"📄 작업 파일 자동 감지: {task_file_path}"
        print(msg)
        log_buffer.append(msg)
    else:
        msg = f"⚠️  작업 파일을 찾을 수 없습니다: {task_file_path}\n   --tasks 인자로 직접 작업을 지정하거나 --task-file로 파일을 지정해야 합니다."
        print(msg)
        log_buffer.append(msg)
    
    # Step 1: physical_guard.py 실행
    success, _ = run_physical_guard(fp_number, task_file=task_file, log_buffer=log_buffer)
    
    if not success:
        error_msg = "\n❌ physical_guard.py 실행이 실패했습니다. evaluate_results.py는 실행하지 않습니다."
        print(error_msg)
        log_buffer.append(error_msg)
        save_log_to_file(log_buffer, start_time)
        sys.exit(1)
    
    # Step 2: evaluate_results.py 실행
    folder = get_folder_from_fp_number(fp_number)
    success, _ = run_evaluate_results(fp_number, folder=folder, log_buffer=log_buffer)
    
    if not success:
        error_msg = "\n⚠️  evaluate_results.py 실행이 실패했습니다."
        print(error_msg)
        log_buffer.append(error_msg)
    
    # 종료 시간 기록
    end_time = datetime.now()
    duration = end_time - start_time
    
    completion_msg = f"\n{'='*60}\n✅ 모든 스크립트 실행 완료!\n{'='*60}\n"
    completion_msg += f"실행 종료 시간: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    completion_msg += f"총 실행 시간: {duration}\n"
    print(completion_msg)
    log_buffer.append(completion_msg)
    
    # 로그 파일 저장
    save_log_to_file(log_buffer, start_time)
    
    if not success:
        sys.exit(1)

def save_log_to_file(log_buffer: list, start_time: datetime):
    """
    로그 버퍼의 내용을 total_log.txt 파일로 저장합니다.
    
    Args:
        log_buffer: 로그를 저장한 리스트
        start_time: 실행 시작 시간
    """
    try:
        script_dir = Path(__file__).parent
        log_file = script_dir.parent / "total_log.txt"
        
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"Unified Execution Log\n")
            f.write(f"생성 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            f.write("\n".join(log_buffer))
            f.write("\n\n" + "="*80 + "\n")
            f.write(f"로그 저장 완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n")
        
        print(f"\n📝 모든 로그가 저장되었습니다: {log_file}")
    except Exception as e:
        print(f"\n⚠️  로그 파일 저장 중 오류 발생: {e}")

if __name__ == "__main__":
    main()
