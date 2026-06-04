-- UPH (EQUIP_UPH) — maps to UphRecord; availability is derived (UPH > 0).
--
-- Bind: :rule_timekey, :fac_id
-- Columns: RULE_TIMEKEY, PLAN_PROD_KEY, OPER_ID, EQP_MODEL_CD, UPH

SELECT RULE_TIMEKEY,
       PLAN_PROD_KEY,
       OPER_ID,
       EQP_MODEL_CD,
       TO_NUMBER(ATTR_VAL) AS UPH
  FROM RTS_LINEDSDB_INF
 WHERE RULE_TIMEKEY = :rule_timekey
   AND FAC_ID = :fac_id
   AND GBN_CD = 'EQUIP_UPH'
