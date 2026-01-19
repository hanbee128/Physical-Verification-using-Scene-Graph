# Scene Graph based Physical Verification

## 개요

`physical_guard.py`는 논리적 검증을 마친 AI2-THOR 플랜을 물리적 검증하는 스크립트입니다. Scene Graph를 활용하여 각 액션의 물리적 사전 조건을 검증하고, 실패한 경우 복구 액션을 자동으로 생성하여 최종 실행 가능한 플랜을 생성합니다.

## 주요 기능

### 1. 논리적 검증 + 물리적 검증 파이프라인

- **논리적 검증**: LLM(Ollama llama3)을 사용하여 자연어 작업 설명으로부터 Python 스타일 프로그램 생성
- **물리적 검증**: Scene Graph를 기반으로 각 액션의 물리적 사전 조건 검증
- **복구 액션 생성**: 검증 실패 시 자동으로 복구 액션 생성 및 삽입
- **Scene Graph 업데이트**: 액션 실행 후 Scene Graph 상태 자동 업데이트

### 2. 지원하는 액션 타입

- `GoToObject(object)`: 객체 근처로 이동
- `PickupObject(object)`: 객체 집기
- `PutObject(object, receptacle)`: 객체를 수용체에 놓기
- `OpenObject(object)`: 객체 열기
- `CloseObject(object)`: 객체 닫기

### 3. 물리적 가드(Guard) 검증

각 액션 타입별로 다음 가드들을 검증합니다:

#### GoToObject
- `EXISTS(object)`: 객체가 Scene Graph에 존재하는지 확인
- `NAVIGABLE(agent, object)`: NavMesh를 통해 객체까지 이동 가능한 경로가 있는지 확인 (거리 1.3m 이내)

#### PickupObject
- `pickupable(object)`: 객체가 pickupable 속성을 가지고 있는지 확인
- `REACHABLE(agent, object)`: Agent의 손 위치 기준으로 객체가 도달 가능한 범위 내에 있는지 확인
- `¬HOLDS(agent, *)`: Agent가 현재 아무 객체도 들고 있지 않은지 확인
- `¬IN(object, closed_receptacle)`: 객체가 닫힌 수용체 안에 있지 않은지 확인

#### PutObject
- `HOLDS(agent, object)`: Agent가 목표 객체를 들고 있는지 확인
- `receptacle(receptacle)`: 수용체가 receptacle 타입인지 확인
- `REACHABLE(agent, receptacle)`: Agent의 손 위치 기준으로 수용체가 도달 가능한 범위 내에 있는지 확인
- `openable(receptacle)`: 수용체가 openable 속성을 가지고 있는지 확인
- `OPENED(receptacle)`: 수용체가 열려있는지 확인 (openable인 경우에만)

#### OpenObject
- `openable(object)`: 객체가 openable 속성을 가지고 있는지 확인
- `¬OPENED(object)`: 객체가 이미 열려있지 않은지 확인
- `REACHABLE(agent, object)`: Agent의 손 위치 기준으로 객체가 도달 가능한 범위 내에 있는지 확인

#### CloseObject
- `openable(object)`: 객체가 openable 속성을 가지고 있는지 확인
- `OPENED(object)`: 객체가 열려있는지 확인
- `REACHABLE(agent, object)`: Agent의 손 위치 기준으로 객체가 도달 가능한 범위 내에 있는지 확인

### 4. REACHABLE 검증 상세

`REACHABLE` 가드는 Agent의 손 위치를 기준으로 객체가 도달 가능한 범위 내에 있는지 확인합니다.

#### 손 위치 계산
- Agent의 절대 좌표를 기준으로 손 위치 계산 (Agent 위치를 손 위치로 직접 사용)
- Agent 위치 = 손 위치 (절대 좌표 기준)

#### 도달 가능 범위
- `x` 범위: Agent 위치 기준 -1.452m ~ +0.793m (좌우)
- `y` 범위: Agent 위치 기준 -0.275m ~ +0.853m (상하)
- `z` 범위: Agent 위치 기준 -1.620m ~ +0.646m (앞뒤)

이 범위는 `use_arm_and_armbase.py`에서 테스트한 실제 로봇팔 도달 범위를 기반으로 설정되었습니다.

**중요**: REACHABLE 가드 위반 시 물리적 검증이 즉시 종료되며, 계획을 생성할 수 없습니다.

### 5. 복구 액션 자동 생성

검증 실패 시 다음 복구 액션들이 자동으로 생성됩니다:

#### PickupObject 실패 시
- `¬IN(object, closed_receptacle)` 실패 → 부모 수용체를 여는 `OpenObject` 액션 추가
- `REACHABLE(agent, object)` 실패 → **검증 종료** (물체가 닿지 않는 거리에 있어 계획 생성 불가)

#### PutObject 실패 시
- `HOLDS(agent, object)` 실패 → `GoToObject` + `PickupObject` 액션 추가
- `OPENED(receptacle)` 실패 → 수용체를 여는 `OpenObject` 액션 추가

#### PickupObject 성공 후
- 부모 수용체가 열려있으면 자동으로 `CloseObject` 액션 추가

#### 복구 액션으로 OpenObject 추가 시
- 복구 액션으로 `OpenObject`가 추가되면, 원래 액션 이후에 자동으로 `CloseObject` 액션 추가

### 6. Scene Graph 업데이트

액션이 성공적으로 검증 통과하면 Scene Graph가 자동으로 업데이트됩니다:

#### GoToObject
- Agent의 위치를 NavMesh에서 찾은 가장 가까운 이동 가능 위치로 업데이트

#### PickupObject
- `HOLDS` 엣지 추가
- `IN` 엣지 제거
- Agent 노드의 `isHolding=True`, `heldObjectId` 설정
- Object 노드의 `isPickedUp=True`, `parentReceptacles` 초기화

#### PutObject
- `HOLDS` 엣지 제거
- `IN` 엣지 추가
- Agent 노드의 `isHolding=False`, `heldObjectId=None`
- Object 노드의 `isPickedUp=False`, `parentReceptacles` 업데이트

#### OpenObject
- Object 노드의 `isOpen=True`, `openness=1.0` 설정

#### CloseObject
- Object 노드의 `isOpen=False`, `openness=0.0` 설정

### 7. Task 완료 검증

모든 액션이 통과하지 못한 경우, LLM을 사용하여 자연어 목표와 최종 플랜을 비교하여 누락된 작업을 찾습니다.

- 누락된 작업이 발견되면 자동으로 해당 작업에 대한 플랜을 생성하고 기존 플랜에 추가
- 누락된 task plan은 `[LLM 생성 - 누락 Task 보완]` 마커로 표시됨

### 8. 검증 종료 조건

다음 조건에서 물리적 검증이 즉시 종료됩니다:

- **EXISTS 가드 실패**: 객체가 Scene Graph에 존재하지 않는 경우
- **REACHABLE 가드 실패**: 물체가 agent의 손이 닿지 않는 거리에 있는 경우

### 9. Plan 출력 형식

최종 플랜은 다음 마커로 구분되어 출력됩니다:

- `[LLM 생성 - 논리적 검증]`: LLM이 논리적 검증 단계에서 생성한 원본 액션
- `[LLM 생성 - 누락 Task 보완]`: LLM이 물리적 검증 후 누락된 task를 위해 추가로 생성한 액션
- `[시스템 생성]`: 시스템이 물리적 검증 단계에서 자동으로 생성한 복구 액션
- `[LLM 주석]`: LLM이 실패한 액션에 대해 생성한 설명 주석

**실패한 액션**: 물리적 검증 실패한 액션은 주석처리되어 출력됩니다.

### 10. Baseline 비교 평가

`evaluation.py`를 사용하여 Baseline(ProgPrompt)와 Physical Guard 결과를 비교할 수 있습니다.

**평가 지표:**
1. **Task Success Rate (작업 성공률)**: 작업 완료율
2. **Action Pass Rate (액션 통과율)**: 물리적 검증 통과율
3. **Recovery Action Effectiveness (복구 액션 효과성)**: 복구 액션 성공률
4. **Plan Executability (계획 실행 가능성)**: 실행 가능한 계획 비율

**사용 방법:**
```bash
python scripts/evaluation.py \
    --baseline-json results/ai2thor_progprompt_*.json \
    --physical-guard-txt results/physical_guard_set3_result_*.txt \
    --output results/evaluation_comparison.txt
```

`physical_guard.py` 실행 시 Baseline 실행 후 자동으로 평가가 수행됩니다.

## 사용 방법

### 기본 사용법

```bash
python scripts/physical_guard.py --scene-number 1 --tasks "put apple in fridge"
```

### 주요 옵션

- `--scene-number`: FloorPlan 번호 지정 (예: 1, 201, 301, 401)
- `--scene-graph`: Scene Graph JSON 파일 경로 직접 지정
- `--tasks`: 작업 목록 (인라인)
- `--task-file`: 작업 목록이 담긴 JSON/JSONL 파일 경로
- `--model`: Ollama 모델 이름 (기본값: llama3)
- `--ollama-url`: Ollama 엔드포인트 URL (기본값: http://localhost:11434/v1)
- `--output-dir`: 결과 저장 디렉토리 (기본값: results)
- `--info-file`: 액션과 객체 정보가 담긴 info.txt 파일 경로

### 예시

```bash
# Scene 번호 1 사용
python scripts/physical_guard.py --scene-number 1 --tasks "put apple in fridge"

# 여러 작업 처리
python scripts/physical_guard.py --scene-number 1 --tasks "put apple in fridge" "put tomato in fridge"

# 작업 파일 사용
python scripts/physical_guard.py --scene-number 1 --task-file tasks.json

# 다른 모델 사용
python scripts/physical_guard.py --scene-number 1 --model llama3.1 --tasks "put apple in fridge"
```

## 출력 파일

### JSON 파일
- 경로: `results/ai2thor_progprompt_{timestamp}.json`
- 형식: `{task: program_code}` 딕셔너리

### 텍스트 파일
- 경로: `results/physical_guard_set3_result_{task_name}_{timestamp}.txt`
- 내용:
  - 작업별 프로그램 코드 (LLM 생성/시스템 생성 구분 표시)
  - 물리적 검증 요약 (통과/실패 액션 수)
  - 실패한 액션 목록 및 이유 (주석처리됨)
  - Task 완료 검증 결과

### 평가 비교 파일
- 경로: `results/evaluation_comparison_{timestamp}.txt`
- 내용: Baseline과 Physical Guard 결과 비교 평가 지표

### Scene Graph 업데이트 파일
- 경로: `scripts/updated_scene_graph.json`
- 각 액션 실행 후 Scene Graph 상태가 업데이트되어 저장됨

## Scene Graph 파일 구조

Scene Graph는 다음 구조를 가집니다:

```json
{
  "nodes": {
    "agent": {
      "position": {"x": float, "y": float, "z": float},
      "rotation": {"x": float, "y": float, "z": float},
      "isHolding": bool,
      "heldObjectId": string | null
    },
    "objects": [
      {
        "nodeId": string,
        "objectType": string,
        "position": {"x": float, "y": float, "z": float},
        "pickupable": bool,
        "openable": bool,
        "receptacle": bool,
        "isOpen": bool,
        "openness": float,
        "isPickedUp": bool,
        "parentReceptacles": [string]
      }
    ]
  },
  "edges": [
    {
      "edgeType": string,
      "source": string,
      "target": string
    }
  ]
}
```

## 검증 프로세스

1. **프로그램 파싱**: LLM이 생성한 프로그램 코드를 액션 리스트로 파싱
2. **액션별 검증**: 각 액션에 대해 Scene Graph 기반 물리적 검증 수행
3. **검증 종료 조건 확인**: EXISTS 또는 REACHABLE 가드 실패 시 즉시 검증 종료
4. **복구 액션 생성**: 검증 실패 시 자동으로 복구 액션 생성 및 삽입
   - 복구 액션으로 `OpenObject` 추가 시, 원래 액션 이후에 `CloseObject` 자동 추가
5. **Scene Graph 업데이트**: 검증 통과한 액션 실행 후 Scene Graph 상태 업데이트
6. **최종 플랜 생성**: 통과한 액션과 실패한 액션(주석처리)을 포함한 최종 플랜 생성
   - LLM 생성/시스템 생성 구분 표시
7. **Task 완료 검증**: LLM을 사용하여 누락된 작업 확인 및 추가 플랜 생성
8. **Baseline 비교 평가**: Baseline 실행 후 자동으로 평가 수행

## 주요 함수

### `parse_program_to_actions(program_code)`
프로그램 코드를 파싱하여 액션 리스트로 변환

### `load_scene_graph(scene_graph_path)`
Scene Graph JSON 파일 로드

### `get_relevant_scene_context(scene_graph, action_type, object_name, receptacle_name)`
Scene Graph에서 관련 노드와 엣지 정보 추출

### `find_closest_reachable_position(controller, target_pos)`
NavMesh에서 목표 위치까지 가장 가까운 이동 가능 위치 찾기

### `verify_guard_with_scene_graph(guard_name, scene_context, ...)`
개별 가드 검증 (EXISTS, REACHABLE, HOLDS, IN, OPENED, PICKUPABLE, RECEPTACLE, OPENABLE, NAVIGABLE)

### `verify_action_with_scene_graph(action, scene_graph, controller)`
액션의 모든 가드 검증 및 복구 액션 생성

### `update_scene_graph_after_action(scene_graph, action, verification_passed, ...)`
액션 실행 후 Scene Graph 업데이트

### `generate_final_plan_with_physical_verification(task, initial_program, scene_graph, ...)`
논리적 검증된 플랜을 물리적 검증하여 최종 플랜 생성

### `verify_task_completion_with_llm(client, model, task, final_plan)`
LLM을 사용하여 자연어 목표와 최종 플랜을 비교하여 누락된 작업 확인

### `generate_failure_comment_with_llm(client, model, action, failed_guards, failure_reason)`
LLM을 사용하여 물리적 검증 실패 이유를 분석하고 주석 생성

### `evaluation.py`
Baseline과 Physical Guard 결과를 비교하여 평가 지표 계산

## 의존성

- `ai2thor`: AI2-THOR 시뮬레이터
- `openai`: OpenAI 호환 API 클라이언트 (Ollama와 통신)
- `scene_graph_extractor`: Scene Graph 추출 모듈 (선택사항)
- `ai2thor_connector`: AI2-THOR 실행 커넥터

## 주의사항

1. **Scene Graph 파일**: Scene Graph 파일이 올바른 경로에 있어야 합니다.
   - 기본 경로: `scripts/scene_graph_structured_FloorPlan{number}.json`
   - `--scene-number` 옵션으로 자동 생성되거나 `--scene-graph`로 직접 지정 가능

2. **Ollama 서버**: Ollama 서버가 실행 중이어야 합니다.
   - 기본 URL: `http://localhost:11434/v1`

3. **AI2-THOR Controller**: NavMesh 검증을 위해서는 AI2-THOR Controller가 초기화되어야 합니다.
   - Controller 초기화 실패 시 NavMesh 검증이 제한됩니다.

4. **최대 검증 횟수**: 무한 루프 방지를 위해 최대 검증 횟수는 20회로 제한됩니다.

## 로그 출력

스크립트는 다음 정보를 로그로 출력합니다:

- 액션별 물리적 검증 결과
- 각 가드의 통과/실패 여부 및 이유
- 복구 액션 생성 정보
- Scene Graph 업데이트 정보
- Task 완료 검증 결과

## 예시 출력

```
[액션 1/5] GoToObject('Apple')
  → 물리적 검증: GoToObject('Apple')
  📊 에이전트 및 목표 객체 정보:
    🤖 에이전트 위치: (1.500, 0.901, 1.500)
    🤖 에이전트 회전: 0.0°
    🤖 손에 들고 있는 객체: None
    📦 목표 객체: Apple
    📦 객체 위치: (2.155, 1.099, 0.617)
    📦 거리: 1.234m
  → 검증할 Guards: ['EXISTS(object)', 'NAVIGABLE(agent, object)']
    ✓ EXISTS(object): 객체 'Apple' 존재 확인
    ✓ NAVIGABLE(agent, object): 가장 가까운 이동 가능 위치까지 거리: 0.85m <= 1.3m
  ✓ 검증 통과: 모든 가드 통과 (2개)
  ✓ Agent 위치 업데이트: GoToObject('Apple') → (2.100, 0.901, 0.600) (거리: 0.850m)
```

## 문제 해결

### Scene Graph 파일을 찾을 수 없음
- `--scene-number` 옵션으로 올바른 번호를 지정했는지 확인
- `scripts/scene_graph_structured_FloorPlan{number}.json` 파일이 존재하는지 확인

### Ollama 연결 실패
- Ollama 서버가 실행 중인지 확인: `curl http://localhost:11434/api/tags`
- `--ollama-url` 옵션으로 올바른 URL을 지정했는지 확인

### NavMesh 검증 실패
- AI2-THOR Controller 초기화가 성공했는지 확인
- Scene 이름이 올바른지 확인 (예: `FloorPlan1_physics`)

## 참고

- 초기 로봇팔 자세: armBase 좌표계에서 `(x=0, y=0, z=0.5)`
- MoveArmBase 초기 y 값: normalized `0.5` (0~1 범위에서 중간 높이)
- REACHABLE 검증은 Agent의 손 위치를 기준으로 절대 좌표로 비교합니다.
- **REACHABLE 가드 위반 시**: 물체가 agent의 손이 닿지 않는 거리에 있어 계획을 생성할 수 없으므로 검증이 즉시 종료됩니다.
- **복구 액션 OpenObject**: 복구 액션으로 수용체를 열면, 원래 액션 실행 후 자동으로 닫힙니다.
- **실패한 액션**: 물리적 검증 실패한 액션은 주석처리되어 실행되지 않습니다.
- **Plan 마커**: LLM이 생성한 부분과 시스템이 생성한 부분이 명확히 구분되어 출력됩니다.

## 평가 지표

`evaluation.py`를 통해 다음 지표들을 계산할 수 있습니다:

1. **Task Success Rate**: 작업 완료율
   - Physical Guard: `successful_tasks / tasks_with_result * 100`
   - Baseline: 물리적 검증 없음 (N/A)

2. **Action Pass Rate**: 액션 통과율
   - `passed_actions / total_actions * 100`

3. **Recovery Action Effectiveness**: 복구 액션 효과성
   - `recovery_actions / failed_actions * 100`

4. **Plan Executability**: 계획 실행 가능성
   - `(total_actions - failed_actions) / total_actions * 100`

```bash
# 직접 실행
python scripts/evaluation.py \
    --baseline-json results/ai2thor_progprompt_20260108-101853.json \
    --physical-guard-txt results/physical_guard_set3_result_Put_Tomato_and_Apple_and_Potato_in_Fridge_20260108-101831.txt \
    --output results/evaluation_comparison.txt

```