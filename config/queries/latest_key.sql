-- The most recent RULE_TIMEKEY (defaults inference target when --timekey is
-- omitted).
--
-- Bind variable: :fac_id
-- Required column: single scalar (MAX value)

SELECT MAX(RULE_TIMEKEY)
  FROM RTS_LINEDSDB_INF
 WHERE GBN_CD = 'AVAIL_WIP_QTY'
   AND FAC_ID = :fac_id
