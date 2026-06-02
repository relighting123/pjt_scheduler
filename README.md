해당 프로젝트는 장비 전환 스케줄링 시스템이다.(파일명에 스냅샹이나 dsdb 등의 용어는 쓰지 말고 일반적 용어로 해줘)

[1] 프로젝트 구성
1. core 폴더가 잇으며 아래에는 db 연결,조회,저장 등 기능/객체 구조시물레이터 , 강화학습 엔진 등 공용 컴포넌트들이 있어.
2. biz영역을 core를 활용한 구체화 부분이야. 정의된 스키마 기반으로 실제 데이터를 가져와 데이터 클래스로 정의해서 이를 core 시물레이터에 매핑하여 구동까지 하는거야.  이런 방향으로 프로젝트 구성이 필요해

[2] 강화학습 모델링
주어진 plan prod key/OPER별 계획량에 대해 장비배치를 수행하여 전체 평균 달성율을 높이는 데 그 목적이 있다.
plan prod key/OPER별 가능한 모델과 uph는 차이가 있으므로 이를 고려하여 최적 배치를 모방학습과 강화학습을 통해 해결해나가고자 한다.

[4] 시스템 세부 정보
해당 비즈니스는 강화학습과 모방학습을 결합한다.
학습 모듈과 추론 모듈이 있다.
강화학습은 Stablebaseline을 활용한다.
강화학습은 모방학습으로 초기 정책을 학습하고 이후 PPO 모델을 통해 학습한다.
학습 후에 지정한 벤치마크 데이터 기반으로 평가하여 어느정도 수준의 모델인지 관리하는 md 파일을 업데이트한다.(기록용)

1. 제품별 공정 수순 정보 및 재공 정보
 RULE_TIMEKEY | PLAN_PROD_KEY  | OPER_ID | OPER_SEQ | WIP_QTY

2. 제품별 공정별 장비 모델별 시간당 생산량
 RULE_TIMEKEY | PLAN_PROD_KEY  | OPER_ID | EQP_MODEL_CD | UPH 

3. 제품별 공정별 장비 모델별 댓수
 RULE_TIMEKEY | BATCH_ID | EQP_MODEL_CD |  EQP_QTY

5. 제품별 공정별 장비 모델별 처리가능여부
 RULE_TIMEKEY | PLAN_PROD_KEY  | OPER_ID | EQP_MODEL_CD | AVAIL_YN

6. Tool 교체 단위 정보
 RULE_TIMEKEY | BATCH_ID | PLAN_PROD_KEY | OPER_ID
Batch id는 plan prod key와 oper id에 의해 정의된다. pla prod key||oper_I와 batch id는 N:1 관계이다.

7. Tool 갯수 정보
 RULE_TIMEKEY | BATCH_ID | EQP_MODEL_CD | TOOL_QTY

8. 계획 정보
 RULE_TIMEKEY | PLAN_PROD_KEY  | OPER_ID | START TIME | END TIME | PLAN_QTY
계획 제품 Key / Oper 별 계획이 있고 세부 일자 시간대별 계획이 있다. 가령 P1 / PT1H / 2026051707 | 2026051708 | 100 이면 2026년 5월 17일 07시부터 08시까지 10000개를 생산하라는 계획이야. 그리고 2026051708 | 2026051807 | 300 이면 2026년 5월 17일 08시부터 18시까지 300개를 생산하라는 계획이야.
이런식으로 동일 제품에 대해 여러 계획이 있을 수 있따.

위 정보는 실제 물리 테이블에서는 [8]영역에 RTS_LINEDSDB_INF 테이블 하나로 관리되며 쿼리를 통해 위 형태로 변경 후 강화학습 모델에 들어간다.


output

RULE_TIMEKEY | FROM_BATCH | FROM_PLAN_PROD_KEY | FROM_OPER_ID | EQP_MODEL_CD | TO_BATCH_ID | TO_PLAN_PROD_KEY | TO_OPER_ID |  EQP_MODEL_CD | START_CONV_TIME | EQP_QTY
형태로 데이털르 생성한다.
2026051723020000 형태가 Rule TImekey이고 현재 수행시간을 의미한다. 테이블 명은 RTD_CONV_INF / RTD_CONV_HIS 이고 삭제후 insert하게 된다.
start time은 rule timekey과 동일 시점이 된다.

[5] 장비 Tool 전환 범위
Tool 전환에 대해 점진적 적용을 위해 특정 batch id 별 그루핑.정의하며 해당 그룹 내에서만 전환하게 한다.
가령 G001 : [9C/92,9C/102] 라면 tool 전환은 9c/92와 9C/102 끼리만 가능하며 나머지는 Tool 전환은 없다 단 batch id가 동일하다면 세부 Plan prod key와 oper 간 전환은 자유롭다

[6] 배경지식
 batch id가 달라지는 경우는 tool 교체가 일어나며 to batch id의 tool은 소진하고 from batch id의 tool은 반환한다. 동일 BATCH_ID 내 PLAN PROD KEY가 변경되는 부분에 대해서는 시간 소요 및 TOOL 교체는 없다.

[7] 학습·추론 RULE_TIMEKEY 운영

**학습**
- `from_rule_timekey` ~ `to_rule_timekey` 구간의 각 스냅샷을 DB에서 조회하여 학습 (구간에 여러 키가 있으면 에피소드마다 무작위 스냅샷).
- 단일 스냅샷만 지정할 때는 `rule_timekey` 또는 `from`/`to`에 동일 값 지정.
- 학습 완료 후 **전체 벤치마크** 데이터셋으로 Optimal·휴리스틱·RL 성능 비교 (`evaluate_all_benchmark_datasets`). 작업을 수행한다.
- 요약 결과에는 벤치마크별로 PLAN PROD KEY / MODEL별 댓수 및 달성률과 평균 달성률이 나오고 실제 최적해 기준 동일 형태로 데이터를 제시한다. 
  또한 메인페이지에는 각 벤치마크별 최적해와 추론 결과 기반으로 평균 계획달성률 및 장비전환 횟수를 제시한다. 이를 html로 제공한다. 그래서 최적 정답과 추론 정답 간 정답 비교를 한다.
- 벤치마크 입력: CSV 7종 + `ground_truth.json` (`python test_benchmark.py`로 DB 없이 검증).
- 벤치 마크 데이터는 무조건 최적 정답을 이미 알고 있는 문제로 만들어야 하며 7개 정도 벤치마크 데이터가 필요하다.
문제의 경우 특정 공정에만 장비가 몰려서 전환을 해야 최종 공정 달성이 가능한 경우 등으로 다양한 문제가 뭐야
**추론**
- `rule_timekey`: 조회 스냅샷·RTD_CONV 등 **결과 출력 키 동일**. 미지정·N/A 시 `MAX(RULE_TIMEKEY)` (WIP_INFO).

**CLI 예시**
```bash
python run.py train --from-timekey 20251020070000 --to-timekey 20251020120000 --steps 50000
python run.py infer --timekey 20251020070000
python run.py infer   # RULE_TIMEKEY=DB MAX (입력·출력 동일)
```

[8]
##대상 DB
오라클 DB -> config 관리

oracle db는 하기 정보를 활용해
서비스 이름 : XEPDB1
IP와 포트 : localhost:1521
계정/암호 : dispatcher/dispatcher

### 데이터 구조
테이블 명 : RTS_LINEDSDB_INF
테이블 스키마
RULE_TIMEKEY VARCHAR2(50) PK
FAC_ID VARCHAR2(50) PK,
BATCH_ID VARCHAR2(50),PK
PLAN_PROD_KEY VARCHAR2(200),PK
OPER_ID  VARCHAR2(50),PK
OPER_SEQ NUMBER,
EQP_MODEL_CD VARCHAR2(50),PK
GBN_CD VARCHAR2(50), PK
ATTR_VAL VARCHAR2(50) 

데이터 값(예시)
RULE_TIMEKEY  : "2026052922500000"
FAC_ID : ["ICPRB","CJPRB"]
BATCH_ID : ["9C/92","9C/102",..]
PLAN_PROD_KEY : ["M15/59C/H5UDGSTED/E1S/NA"]
OPER_ID : ["Z1020000A","Z1040000A"]
OPER_SEQ : 1,2,3..
EQP_MODEL_CD : ["T5833","MAGNUM5"]
GBN_CD : ["ASSIGN_EQUIP_CNT","UPH","WIP_QTY","D0_TARGET_QTY","D1_TARGET_QTY","TOOL_QTY"]
ATTR_VAL : 각 항목별 GBN_CD에 해당하는 값

**만약 PLAN_PROD_KEY/EQP_MODEL 기준 조회시 UPH가 없다면 진행 불가로 판단함.
**D0_TARGET의 경우 RULE_TIMEKEY 기준에서 다음날 07시까지의 계획이며 D1 TARGET의 경우 다음날 07시에서 그 다음날 07시까지 계획으로 치환하여 처리

**API parameters (rl_train)**
- `from_rule_timekey`, `to_rule_timekey`, `rule_timekey`, `run_test_eval`, `benchmark_dataset`

**API parameters (rl_inference)**
- `rule_timekey` (task 또는 parameters) — 조회·출력 공통

## Created Project Usage

```bash
python run.py eval
python run.py train --benchmark-dataset benchmarks/benchmark_01 --steps 50000
python run.py infer --benchmark-dataset benchmarks/benchmark_01 --output artifacts/inference/allocation.csv
```

- `core`: domain model, simulator, optimizer, evaluation, optional RL training interface.
- `biz`: Oracle/config adapters that map real tables into core datasets and persist output tables.
- `benchmarks`: 11 CSV benchmark datasets plus `ground_truth.json` for DB-free validation.
- `config/settings.json`: Oracle connection and output table/model artifact settings.

Optional packages:

```bash
pip install -e .[rl,oracle]
```

### WIP handling (단일 스냅샷)

각 OPER의 생산량은 `WIP_QTY`를 상한으로 캡된다 — 재공이 부족하면 장비가 남아도
그 이상 생산하지 못한다. 단일 스냅샷 경로에서 WIP=0/미기록은 하위호환을 위해
"무제한"으로 간주한다 (`benchmark_11`이 OP20 재공 50개 한계를 검증).

### 멀티 피리어드 (WIP 흐름) — `core/flow.py`

단일 스냅샷은 "지금 충분한 재공"을 가정한다. 실제로는 하위 공정 큐가 비어있어
**앞 공정에서 재공을 먼저 쌓고(build-ahead) → 전환 → 뒤 공정 처리**가 필요하다.
`MultiPeriodSimulator`는 horizon을 슬롯으로 쪼개고, 각 슬롯의 생산이 다음 공정
(OPER_SEQ 순)의 재공으로 **다음 슬롯에** 흘러가도록 모델링한다. 슬롯마다 정책이
현재 재공/잔여계획을 보고 재배치하며, 배치를 넘으면 전환으로 집계된다.

```bash
python test_multiperiod.py   # DB 없이 build-ahead 시나리오 검증
```

- `static` (단일 배치 고정) 0.5  vs  `dynamic`(슬롯별 재배치) 1.0  vs  `optimal` 1.0
- 정책: `static_policy`, `dynamic_greedy_policy`, 소규모 정확해 `multiperiod_optimal`.

