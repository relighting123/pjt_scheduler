-- Insert a single conversion row into the latest-result table. Executed via
-- executemany — one row per Allocation diff entry. Bind names below match
-- biz/output_writer.OUTPUT_COLUMNS.

INSERT INTO RTD_CONV_INF (
    RULE_TIMEKEY,
    FROM_BATCH,
    FROM_PLAN_PROD_KEY,
    FROM_OPER_ID,
    FROM_EQP_MODEL_CD,
    TO_BATCH_ID,
    TO_PLAN_PROD_KEY,
    TO_OPER_ID,
    TO_EQP_MODEL_CD,
    START_CONV_TIME,
    EQP_QTY
) VALUES (
    :rule_timekey,
    :from_batch,
    :from_plan_prod_key,
    :from_oper_id,
    :from_eqp_model_cd,
    :to_batch_id,
    :to_plan_prod_key,
    :to_oper_id,
    :to_eqp_model_cd,
    :start_conv_time,
    :eqp_qty
)
