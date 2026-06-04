-- Equipment pool (ASSIGN_EQUIP_CNT) — maps to EquipmentRecord (summed per batch/model).
--
-- Bind: :rule_timekey, :fac_id
-- Columns: RULE_TIMEKEY, BATCH_ID, EQP_MODEL_CD, EQP_QTY

SELECT RULE_TIMEKEY,
       BATCH_ID,
       EQP_MODEL_CD,
       TO_NUMBER(ATTR_VAL) AS EQP_QTY
  FROM RTS_LINEDSDB_INF
 WHERE RULE_TIMEKEY = :rule_timekey
   AND FAC_ID = :fac_id
   AND GBN_CD = 'ASSIGN_EQUIP_CNT'
