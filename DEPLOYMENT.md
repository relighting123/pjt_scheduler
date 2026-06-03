# 회사 환경 배포 · 이식 절차서

장비 전환 스케줄러를 회사 Oracle/MES 환경에 올릴 때 따르는 단계별 가이드.

**설계 요약 (입출력 데이터)**  
- `config/settings.json`에는 **DB 접속 정보·모델·라인 설정만** 둔다.  
  테이블명·SELECT/INSERT 문은 **넣지 않는다**.  
- 입력 스냅샷·키 목록·최신 키·결과 DELETE/INSERT는 모두  
  `config/queries/*.sql` (또는 라인별 복사본)에서 **직접 SQL로 정의**한다.  
- Python(`biz/data_loader.py`, `biz/output_writer.py`)은 SQL 파일을 읽어  
  bind 변수만 넘기고, 피벗·저장 로직은 그대로 유지한다.

**역할 분담**

| 역할 | 담당 |
|---|---|
| DBA / 데이터 | §2 스키마·권한, §3-1 SQL 파일(뷰/테이블명·FAC 필터) |
| ML / 스케줄러 | §1 환경, §5 학습·추론, §0 차원 상한 |
| 운영 / MES | §5 cron·Airflow, §7 검증, 결과 테이블 모니터링 |

---

## 이식 절차 요약 (체크리스트)

| 단계 | 내용 | 완료 |
|:---:|---|:---:|
| 1 | 저장소 clone, `pip install -e .[rl,oracle]`, DB 없이 테스트 3종 통과 (§1) | ☐ |
| 2 | 입력·출력 Oracle 객체 생성 및 GRANT (§2) | ☐ |
| 3 | `settings.json`에 접속·`query_dir`·`tool_groups`만 반영 (§3) | ☐ |
| 4 | `config/queries/*.sql` 6개를 회사 뷰/테이블명에 맞게 수정 (§3-1) | ☐ |
| 5 | §4 연결·`latest_key`·`source` 1건 조회 확인 | ☐ |
| 6 | 구간 학습 → `artifacts/models/*.zip` 생성 (§5) | ☐ |
| 7 | 추론 1회 → 출력 테이블·이력 확인 (§5, §7) | ☐ |
| 8 | 운영 스케줄 등록 (§5) | ☐ |

구버전 설정(`oracle.source_table` / `output_table` / `history_table`)을 쓰던 경우:  
해당 키는 **제거**하고, 동일 테이블을 가리키도록 **각 SQL 파일의 FROM/INTO 절만** 수정하면 된다.

---

## 0. 사전 점검 — 라인 사이즈 vs RL env 상한

RL 환경(`core/rl/env.py`, `core/rl/env_mp.py`)의 고정 상한:

| 차원 | 상한 | 의미 |
|---|---|---|
| `MAX_BUCKETS` | 16 | `(batch_id, eqp_model_cd)` 풀 종류 수 |
| `MAX_TARGETS` | 32 | `(plan_prod_key, oper_id)` 계획 항목 수 |
| bucket당 장비 수 | 무제한 | env 안에서는 ASSIGN_EQUIP_CNT 그대로 사용 |
| 액션 공간 | 528 | `16 × (32 + 1)`, NO-OP 포함 |

운영 라인이 위 상한을 넘으면 **silent truncation** — 앞에서부터 잘려나가
일부 batch/모델/계획이 무시됨. 운영 전 반드시 측정:

```sql
WITH latest AS (SELECT MAX(RULE_TIMEKEY) AS rk FROM RTS_LINEDSDB_INF)
SELECT
  COUNT(DISTINCT BATCH_ID || '|' || EQP_MODEL_CD) AS n_buckets,
  COUNT(DISTINCT PLAN_PROD_KEY || '|' || OPER_ID) AS n_targets
FROM RTS_LINEDSDB_INF, latest
WHERE RULE_TIMEKEY = latest.rk;
```

- `n_buckets > 16` 또는 `n_targets > 32`이면 `core/rl/env.py`와
  `core/rl/env_mp.py`의 클래스 상수 `MAX_BUCKETS` / `MAX_TARGETS`를
  실측값 + 여유분(예: 1.5×)으로 올린다.
- 변경 후 기존 `.zip` 모델은 obs/action 차원이 달라지므로 **재학습 필수**.

---

## 1. 환경 준비

```bash
git clone <repo-url> pjt_scheduler && cd pjt_scheduler

# Python 3.10+ 권장
pip install -e .[rl,oracle]
# = numpy, pandas, jinja2, gymnasium, stable-baselines3, torch,
#   sb3-contrib, oracledb

# DB 없이 작동 확인 (모두 OK여야 함)
python3 test_benchmark.py
python3 test_multiperiod.py
python3 test_queries.py
python3 run.py eval --mode all
```

---

## 2. Oracle DB 스키마

### 2-1. 입력 테이블 — `RTS_LINEDSDB_INF`

회사 ETL/소스에서 미리 채워주는 테이블 (시스템은 SELECT만).

```sql
CREATE TABLE RTS_LINEDSDB_INF (
  RULE_TIMEKEY   VARCHAR2(50)  NOT NULL,
  FAC_ID         VARCHAR2(50)  NOT NULL,
  BATCH_ID       VARCHAR2(50)  NOT NULL,
  PLAN_PROD_KEY  VARCHAR2(200) NOT NULL,
  OPER_ID        VARCHAR2(50)  NOT NULL,
  OPER_SEQ       NUMBER,
  EQP_MODEL_CD   VARCHAR2(50)  NOT NULL,
  GBN_CD         VARCHAR2(50)  NOT NULL,
  ATTR_VAL       VARCHAR2(50),
  PRIMARY KEY (RULE_TIMEKEY, FAC_ID, BATCH_ID, PLAN_PROD_KEY,
               OPER_ID, EQP_MODEL_CD, GBN_CD)
);

CREATE INDEX IDX_RTS_LINEDSDB_RK ON RTS_LINEDSDB_INF (RULE_TIMEKEY);
```

`biz/data_loader.py`의 `_rows_to_problem`이 해석하는 `GBN_CD` 값:

| GBN_CD | 의미 | 매핑 |
|---|---|---|
| `WIP_QTY` | 현재 재공 | `WipRecord` |
| `UPH` | 시간당 생산량 | `UphRecord` |
| `ASSIGN_EQUIP_CNT` | (배치, 모델)별 장비 보유 수 | `EquipmentRecord` |
| `D0_TARGET_QTY` | 당일~다음날 07시 계획 | `PlanRecord` (합산) |
| `D1_TARGET_QTY` | 다음날 07시~그 다음날 07시 계획 | `PlanRecord` (합산) |
| `TOOL_QTY` | Tool 보유 수 | `ToolQtyRecord` |

### 2-2. 출력 테이블 — `RTD_CONV_INF` / `RTD_CONV_HIS`

`biz/output_writer.py`의 `OUTPUT_COLUMNS`와 컬럼/순서 일치 필수.

```sql
-- 최신 결과 (DELETE + INSERT, 동일 RULE_TIMEKEY 단위로 교체)
CREATE TABLE RTD_CONV_INF (
  RULE_TIMEKEY       VARCHAR2(50)  NOT NULL,
  FROM_BATCH         VARCHAR2(50),
  FROM_PLAN_PROD_KEY VARCHAR2(200),
  FROM_OPER_ID       VARCHAR2(50),
  FROM_EQP_MODEL_CD  VARCHAR2(50),
  TO_BATCH_ID        VARCHAR2(50),
  TO_PLAN_PROD_KEY   VARCHAR2(200),
  TO_OPER_ID         VARCHAR2(50),
  TO_EQP_MODEL_CD    VARCHAR2(50),
  START_CONV_TIME    VARCHAR2(50),
  EQP_QTY            NUMBER
);
CREATE INDEX IDX_RTD_CONV_INF_RK ON RTD_CONV_INF (RULE_TIMEKEY);

-- 이력 (append-only)
CREATE TABLE RTD_CONV_HIS (
  RULE_TIMEKEY       VARCHAR2(50)  NOT NULL,
  FROM_BATCH         VARCHAR2(50),
  FROM_PLAN_PROD_KEY VARCHAR2(200),
  FROM_OPER_ID       VARCHAR2(50),
  FROM_EQP_MODEL_CD  VARCHAR2(50),
  TO_BATCH_ID        VARCHAR2(50),
  TO_PLAN_PROD_KEY   VARCHAR2(200),
  TO_OPER_ID         VARCHAR2(50),
  TO_EQP_MODEL_CD    VARCHAR2(50),
  START_CONV_TIME    VARCHAR2(50),
  EQP_QTY            NUMBER
);
CREATE INDEX IDX_RTD_CONV_HIS_RK ON RTD_CONV_HIS (RULE_TIMEKEY);
```

### 2-3. 권한

```sql
GRANT SELECT ON RTS_LINEDSDB_INF TO <운영계정>;
GRANT SELECT, INSERT, DELETE ON RTD_CONV_INF TO <운영계정>;
GRANT SELECT, INSERT ON RTD_CONV_HIS TO <운영계정>;
```

---

## 3. `config/settings.json` 수정

회사 DB 정보로 `oracle` 섹션만 바꾸면 됨. tool_groups는 라인의 실제 batch
그룹으로 정의. 입출력 SQL은 settings에 박지 않고 별도 파일로 관리 (§3-1).

```json
{
  "oracle": {
    "user":          "<회사 계정>",
    "password":      "<회사 비밀번호>",
    "dsn":           "<호스트>:<포트>/<서비스명>",
    "query_dir":     "config/queries",
    "write_history": true
  },
  "tool_groups": {
    "G001": ["<배치1>", "<배치2>"]
  },
  "dynamic": { "num_slots": 4, "slot_hours": 1.0, "switch_time_hours": 0.0 }
}
```

**보안 권장**: 비밀번호는 운영에서 환경 변수로 빼고 launcher 스크립트가
주입. 현재 코드는 평문 JSON 직접 읽음.

### 3-1. 입출력 SQL — `config/queries/*.sql`

6개의 SQL 파일을 그대로 두거나 회사 환경(테이블/뷰명, 필터, 파티션 힌트
등)에 맞게 자유롭게 편집. settings에는 SQL이 한 줄도 없음.

**입력**:

| 파일 | 역할 | bind | 반환 컬럼 |
|---|---|---|---|
| `source.sql`     | RULE_TIMEKEY 1개 스냅샷 피벗 | `:rule_timekey` | 8개 (아래 순서) |
| `range_keys.sql` | [from_key, to_key] 구간 RULE_TIMEKEY 목록 | `:from_key`, `:to_key` | `RULE_TIMEKEY` 1개 |
| `latest_key.sql` | MAX(RULE_TIMEKEY) | 없음 | scalar |

`source.sql`은 반드시 다음 8개 컬럼을 **이 순서**로 반환:
```
RULE_TIMEKEY, BATCH_ID, PLAN_PROD_KEY, OPER_ID, OPER_SEQ,
EQP_MODEL_CD, GBN_CD, ATTR_VAL
```

**출력**:

| 파일 | 역할 | bind |
|---|---|---|
| `delete_output.sql`  | 출력 테이블에서 현재 RULE_TIMEKEY rows 삭제 | `:rule_timekey` |
| `insert_output.sql`  | 출력 테이블에 신규 conversion row INSERT | 11개 (아래) |
| `insert_history.sql` | 이력 테이블에 conversion row INSERT | 동일 |

`insert_*.sql`의 bind 이름 (소문자 컬럼명, executemany로 row 단위 전송):
```
:rule_timekey, :from_batch, :from_plan_prod_key, :from_oper_id,
:from_eqp_model_cd, :to_batch_id, :to_plan_prod_key, :to_oper_id,
:to_eqp_model_cd, :start_conv_time, :eqp_qty
```

**예 — FAC_ID 필터 + 회사 명명 규칙 적용**:
```sql
-- source.sql
SELECT RULE_TIMEKEY, BATCH_ID, PLAN_PROD_KEY, OPER_ID, OPER_SEQ,
       EQP_MODEL_CD, GBN_CD, ATTR_VAL
  FROM MY_SNAPSHOT_VIEW
 WHERE RULE_TIMEKEY = :rule_timekey
   AND FAC_ID = 'ICPRB';

-- delete_output.sql
DELETE FROM MY_RESULT_TABLE WHERE RULE_TIMEKEY = :rule_timekey;

-- insert_output.sql
INSERT INTO MY_RESULT_TABLE (...) VALUES (:rule_timekey, :from_batch, ...);
```

**자주 쓰는 옵션**:
- `oracle.query_dir`을 라인별로 분리 (`config/queries_icprb/`, `config/queries_cjprb/`).
- `oracle.write_history=false`로 이력 INSERT 끄기.
- 6개 파일 중 일부는 그대로 두고 일부만 편집해도 됨.

---

## 4. 연결 점검

`latest_key.sql` / `source.sql`이 회사 DB에서 실행되는지 확인한다.

```bash
python3 -c "
from biz.pipeline import load_settings, _connect
from biz.data_loader import latest_rule_timekey, load_problem_from_oracle
s = load_settings('config/settings.json')
qd = s['oracle']['query_dir']
conn = _connect(s)
rk = latest_rule_timekey(conn, qd)
print('latest RULE_TIMEKEY:', rk)
if rk:
    p = load_problem_from_oracle(conn, qd, rk, s.get('tool_groups', {}))
    print('wip rows:', len(p.wip), 'uph rows:', len(p.uph))
conn.close()
"
```

→ 최신 키와 wip/uph 건수가 0이 아니면 연결·권한·`source.sql`·데이터가 정상.

---

## 5. 학습 + 추론

```bash
# A. 학습 (예: 1주일 구간)
python3 run.py train --mode wip-static \
  --from-timekey 20251020000000 --to-timekey 20251027000000 \
  --steps 50000

# B. 추론 — DB MAX RULE_TIMEKEY 기준, 결과는 RTD_CONV_INF/HIS에 기록
python3 run.py infer --mode wip-static

# C. 특정 키 추론
python3 run.py infer --mode wip-static --timekey 20251027060000

# D. 멀티 피리어드(시간 인과 + 전환 비용) 운영
python3 run.py train --mode dynamic --from-timekey ... --to-timekey ...
python3 run.py infer --mode dynamic
```

운영 자동화는 cron/Airflow 등에서 `python3 run.py infer --mode <mode>`를
주기 호출. 매번 DB 최신 RULE_TIMEKEY를 기준으로 RTD_CONV_INF 갱신.

각 모드 전체 명령은 `README.md`의 "모델 모드" 섹션 참고.

---

## 6. 운영 주의사항 / 알려진 한계

1. **차원 절단** — 라인 사이즈가 §0의 상한을 넘으면 silent truncation.
   배포 전 반드시 측정 + 필요 시 상한 상향 + 재학습.
2. **WIP 의미** — `wip-static` 모드에서 WIP=0/누락은 "무제한"으로 해석됨
   (`treat_zero_as_unlimited=True`, `core/heuristic.py`).
   실제 빈 큐를 표현하려면 `dynamic` 모드 사용.
3. **벤치마크 한계** — `benchmarks/` 11개는 데모용. 실제 일반화는
   회사 DB의 다양한 스냅샷 구간으로 학습해야 가능.
4. **모델 재학습 트리거** — `MAX_BUCKETS`/`MAX_TARGETS` 변경, obs 구조
   변경, `core/rl/env*.py` 수정 시 기존 `.zip` 모델 호환 불가.

---

## 7. 배포 후 검증 체크리스트

- [ ] `python3 test_queries.py` — 6개 SQL 파일 존재·비어 있지 않음
- [ ] `python3 run.py eval --mode all` — 11 벤치마크 평가 통과 (DB 없이)
- [ ] §4 연결 점검 스크립트로 `latest_rule_timekey` 및 wip/uph 건수 확인
- [ ] 짧은 학습 시도: `python3 run.py train --mode wip-static
      --from-timekey <전날> --to-timekey <오늘> --steps 5000`
      → `artifacts/models/ppo_dispatch_wip_static.zip` 생성
- [ ] 추론: `python3 run.py infer --mode wip-static`
      → `RTD_CONV_INF`(또는 `insert_output.sql` 대상 테이블)에 rows 확인
- [ ] 동일 키 추론 2회 → row count 변함 없어야 함 (replace 패턴)
- [ ] `RTD_CONV_HIS` count는 추론마다 누적 증가 (`write_history=true`일 때)

---

## 8. 라인별 SQL 디렉터리 분리 (선택)

여러 FAB/라인을 한 저장소로 운영할 때:

```text
config/
  settings.json          # oracle.query_dir 만 라인별로 바꿔 실행
  queries/               # 기본(공통) 템플릿
  queries_line_a/        # 라인 A용 source/range_keys/...
  queries_line_b/
```

실행 예:

```bash
# settings에 query_dir을 바꾸지 않고, 런처에서 덮어쓰기 (예시)
export SCHEDULER_QUERY_DIR=config/queries_line_a
# 또는 settings.json 복사본 per line
python3 run.py infer --settings config/settings_line_a.json
```

`--settings` 인자로 라인별 JSON을 지정할 수 있다 (`run.py` 참고).
