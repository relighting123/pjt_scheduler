-- Delete the previous result rows for a single RULE_TIMEKEY before inserting
-- the latest decision. Together with `insert_output.sql` this implements the
-- "replace by RULE_TIMEKEY" pattern used by inference.
--
-- Bind variable: :rule_timekey

DELETE FROM RTD_CONV_INF
 WHERE RULE_TIMEKEY = :rule_timekey
