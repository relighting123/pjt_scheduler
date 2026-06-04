-- Tool quantity (TOOL_QTY) — maps to ToolQtyRecord.
--
-- Bind: :rule_timekey, :fac_id
-- Columns: RULE_TIMEKEY, BATCH_ID, EQP_MODEL_CD, TOOL_QTY

SELECT RULE_TIMEKEY,
       BATCH_ID,
       EQP_MODEL_CD,
       TO_NUMBER(ATTR_VAL) AS TOOL_QTY
  FROM RTS_LINEDSDB_INF
 WHERE RULE_TIMEKEY = :rule_timekey
   AND FAC_ID = :fac_id
   AND GBN_CD = 'TOOL_QTY'
