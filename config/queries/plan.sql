-- Plan (D0_PLAN + D1_PLAN) — maps to PlanRecord; quantities are summed per (pk, op) in Python.
--
-- Bind: :rule_timekey, :fac_id
-- Columns: RULE_TIMEKEY, PLAN_PROD_KEY, OPER_ID, START_TIME, END_TIME, PLAN_QTY

SELECT RULE_TIMEKEY,
       PLAN_PROD_KEY,
       OPER_ID,
       RULE_TIMEKEY AS START_TIME,
       RULE_TIMEKEY AS END_TIME,
       TO_NUMBER(ATTR_VAL) AS PLAN_QTY
  FROM RTS_LINEDSDB_INF
 WHERE RULE_TIMEKEY = :rule_timekey
   AND FAC_ID = :fac_id
   AND GBN_CD IN ('D0_PLAN', 'D1_PLAN')
