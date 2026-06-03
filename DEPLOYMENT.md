# 회사 환경 배포 가이드

운영 라인의 Oracle DB 스키마만 맞추면 바로 사용할 수 있도록 정리.

---

## 0. 사전 점검 — 라인 사이즈 vs RL env 상한

RL 환경(`core/rl_env.py`, `core/rl_env_mp.py`)의 고정 상한:

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

- `n_buckets > 16` 또는 `n_targets > 32`이면 `core/rl_env.py`와
  `core/rl_env_mp.py`의 클래스 상수 `MAX_BUCKETS` / `MAX_TARGETS`를
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
python test_benchmark.py
python test_multiperiod.py
python run.py eval --mode all
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
그룹으로 정의.

```json
{
  "oracle": {
    "user":          "<회사 계정>",
    "password":      "<회사 비밀번호>",
    "dsn":           "<호스트>:<포트>/<서비스명>",
    "source_table":  "RTS_LINEDSDB_INF",
    "output_table":  "RTD_CONV_INF",
    "history_table": "RTD_CONV_HIS"
  },
  "tool_groups": {
    "G001": ["<배치1>", "<배치2>"]
  },
  "dynamic": { "num_slots": 4, "slot_hours": 1.0, "switch_time_hours": 0.0 }
}
```

**보안 권장**: 비밀번호는 운영에서 환경 변수로 빼고 launcher 스크립트가
주입. 현재 코드는 평문 JSON 직접 읽음.

### 3-1. 사용자 정의 입력 쿼리 (선택)

기본은 `source_table` 한 테이블을 직접 SELECT. 회사 환경에 따라 뷰/조인/
파티션 필터 등이 필요하면 **사용자가 작성한 SQL을 그대로 바인딩**해서
쓸 수 있다. `oracle` 섹션에 다음 세 키를 추가:

```json
{
  "oracle": {
    "user": "...",
    "password": "...",
    "dsn": "...",
    "source_table": "",
    "output_table": "RTD_CONV_INF",
    "history_table": "RTD_CONV_HIS",

    "source_query": "SELECT RULE_TIMEKEY, BATCH_ID, PLAN_PROD_KEY, OPER_ID, OPER_SEQ, EQP_MODEL_CD, GBN_CD, ATTR_VAL FROM MY_PIVOT_VIEW WHERE RULE_TIMEKEY = :rule_timekey AND FAC_ID = 'ICPRB'",

    "range_keys_query": "SELECT DISTINCT RULE_TIMEKEY FROM MY_PIVOT_VIEW WHERE RULE_TIMEKEY BETWEEN :from_key AND :to_key AND FAC_ID = 'ICPRB' ORDER BY RULE_TIMEKEY",

    "latest_key_query": "SELECT MAX(RULE_TIMEKEY) FROM MY_PIVOT_VIEW WHERE FAC_ID = 'ICPRB' AND GBN_CD = 'WIP_QTY'"
  }
}
```

규칙:

- 세 키는 **선택**. 미지정/`null`이면 기본 SELECT 사용 (=`source_table` 직접).
- 세 키를 정의하면 `source_table` 값은 무시되어도 무방 (빈 문자열 OK).
- **bind 변수명 고정**: `:rule_timekey`, `:from_key`, `:to_key`를 그대로 사용.
- `source_query`는 반드시 다음 8개 컬럼을 **이 순서**로 반환:
    `RULE_TIMEKEY, BATCH_ID, PLAN_PROD_KEY, OPER_ID, OPER_SEQ,
     EQP_MODEL_CD, GBN_CD, ATTR_VAL`
- `range_keys_query`는 `RULE_TIMEKEY` 단일 컬럼 반환.
- `latest_key_query`는 단일 값 (MAX) 반환.
- 세 쿼리 중 일부만 override 가능 (예: source_query만 정의하고
  range/latest는 기본 사용).

전형적인 활용:

- FAC_ID/CO_DIV 등으로 다중 라인을 한 테이블에 두고 라인별 필터.
- 여러 ETL 테이블을 JOIN해서 캐노니컬 8컬럼 형태로 만들어 공급.
- 파티션 힌트, MATERIALIZED VIEW 사용 등 성능 튜닝.

---

## 4. 연결 점검

```bash
python -c "
from biz.pipeline import load_settings, _connect
from biz.data_loader import latest_rule_timekey
s = load_settings('config/settings.json')
conn = _connect(s)
print('latest RULE_TIMEKEY:', latest_rule_timekey(conn, s['oracle']['source_table']))
conn.close()
"
```

→ 최신 키가 출력되면 연결/권한/데이터 모두 정상.

---

## 5. 학습 + 추론

```bash
# A. 학습 (예: 1주일 구간)
python run.py train --mode wip-static \
  --from-timekey 20251020000000 --to-timekey 20251027000000 \
  --steps 50000

# B. 추론 — DB MAX RULE_TIMEKEY 기준, 결과는 RTD_CONV_INF/HIS에 기록
python run.py infer --mode wip-static

# C. 특정 키 추론
python run.py infer --mode wip-static --timekey 20251027060000

# D. 멀티 피리어드(시간 인과 + 전환 비용) 운영
python run.py train --mode dynamic --from-timekey ... --to-timekey ...
python run.py infer --mode dynamic
```

운영 자동화는 cron/Airflow 등에서 `python run.py infer --mode <mode>`를
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
   변경, `core/rl_env*.py` 수정 시 기존 `.zip` 모델 호환 불가.

---

## 7. 배포 후 검증 체크리스트

- [ ] `python run.py eval --mode all` — 11 벤치마크 평가 통과 (DB 없이)
- [ ] §4 연결 점검 스크립트로 latest_rule_timekey 출력 확인
- [ ] 짧은 학습 시도: `python run.py train --mode wip-static
      --from-timekey <전날> --to-timekey <오늘> --steps 5000`
      → `artifacts/models/ppo_dispatch_wip_static.zip` 생성
- [ ] 추론: `python run.py infer --mode wip-static`
      → `RTD_CONV_INF`에 rows 확인
- [ ] 동일 키 추론 2회 → row count 변함 없어야 함 (replace 패턴)
- [ ] `RTD_CONV_HIS` count는 추론마다 누적 증가
