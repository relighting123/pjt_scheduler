-- Distinct RULE_TIMEKEY values within a [from_key, to_key] range.
-- Used by training to enumerate snapshots in a time window.
--
-- Bind variables: :from_key, :to_key, :fac_id
-- Required columns: RULE_TIMEKEY (single)

SELECT DISTINCT RULE_TIMEKEY
  FROM RTS_LINEDSDB_INF
 WHERE RULE_TIMEKEY BETWEEN :from_key AND :to_key
   AND FAC_ID = :fac_id
 ORDER BY RULE_TIMEKEY
