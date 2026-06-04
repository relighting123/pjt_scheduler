-- Per-RULE_TIMEKEY snapshot pivot for the scheduler input loader.
--
-- Bind variables: :rule_timekey, :fac_id
-- Required columns (in this exact order):
--   RULE_TIMEKEY, BATCH_ID, PLAN_PROD_KEY, OPER_ID, OPER_SEQ,
--   EQP_MODEL_CD, GBN_CD, ATTR_VAL
--
-- Customize freely (FAC_ID/CO_DIV filter, view indirection, joins...)
-- as long as the column list and the bind name stay the same.

SELECT RULE_TIMEKEY, BATCH_ID, PLAN_PROD_KEY, OPER_ID, OPER_SEQ,
       EQP_MODEL_CD, GBN_CD, ATTR_VAL
  FROM RTS_LINEDSDB_INF
 WHERE RULE_TIMEKEY = :rule_timekey
   AND FAC_ID = :fac_id
