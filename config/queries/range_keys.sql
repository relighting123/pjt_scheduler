-- Distinct RULE_TIMEKEY values within a [from_key, to_key] range.
-- Used by training to enumerate snapshots in a time window.
--
-- Bind variables: :from_key, :to_key
-- Required columns: RULE_TIMEKEY (single)

SELECT DISTINCT RULE_TIMEKEY
  FROM RTS_LINEDSDB_INF
 WHERE RULE_TIMEKEY BETWEEN :from_key AND :to_key
   AND FAC_ID = 'CJPRB'
 ORDER BY RULE_TIMEKEY
