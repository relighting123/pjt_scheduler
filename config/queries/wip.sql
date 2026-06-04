-- WIP (AVAIL_WIP_QTY) — maps to WipRecord + tool_group batch mapping.
--
-- Bind: :rule_timekey, :fac_id
-- Columns (order): RULE_TIMEKEY, BATCH_ID, PLAN_PROD_KEY, OPER_ID, OPER_SEQ, WIP_QTY

SELECT RULE_TIMEKEY,
       BATCH_ID,
       PLAN_PROD_KEY,
       OPER_ID,
       OPER_SEQ,
       TO_NUMBER(ATTR_VAL) AS WIP_QTY
  FROM RTS_LINEDSDB_INF
 WHERE RULE_TIMEKEY = :rule_timekey
   AND FAC_ID = :fac_id
   AND GBN_CD = 'AVAIL_WIP_QTY'
