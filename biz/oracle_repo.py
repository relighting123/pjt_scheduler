"""Oracle DB access for line data and conversion output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from biz.mapper import rows_to_dataset
from core.domain import ConversionRecord, SchedulingDataset


def load_settings() -> dict[str, Any]:
    path = Path("config/settings.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _connect():
    try:
        import oracledb
    except ImportError as exc:
        raise RuntimeError("Install oracle support: pip install -e .[oracle]") from exc

    cfg = load_settings()["oracle"]
    dsn = oracledb.makedsn(cfg["host"], cfg["port"], service_name=cfg["service_name"])
    return oracledb.connect(user=cfg["user"], password=cfg["password"], dsn=dsn)


def fetch_rows_for_timekey(rule_timekey: str) -> list[dict[str, Any]]:
    cfg = load_settings()["oracle"]
    table = cfg["source_table"]
    sql = f"""
        SELECT RULE_TIMEKEY, FAC_ID, BATCH_ID, PLAN_PROD_KEY, OPER_ID, OPER_SEQ,
               EQP_MODEL_CD, GBN_CD, ATTR_VAL
        FROM {table}
        WHERE RULE_TIMEKEY = :rtk
    """
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute(sql, rtk=rule_timekey)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        conn.close()


def fetch_timekeys_in_range(from_key: str, to_key: str) -> list[str]:
    cfg = load_settings()["oracle"]
    table = cfg["source_table"]
    sql = f"""
        SELECT DISTINCT RULE_TIMEKEY FROM {table}
        WHERE RULE_TIMEKEY BETWEEN :f AND :t
        ORDER BY RULE_TIMEKEY
    """
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute(sql, f=from_key, t=to_key)
        return [str(r[0]) for r in cur.fetchall()]
    finally:
        conn.close()


def fetch_max_timekey() -> str:
    cfg = load_settings()["oracle"]
    wip_table = cfg.get("wip_info_table", "WIP_INFO")
    sql = f"SELECT MAX(RULE_TIMEKEY) FROM {wip_table}"
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute(sql)
        row = cur.fetchone()
        return str(row[0]) if row and row[0] else ""
    finally:
        conn.close()


def load_dataset_from_db(rule_timekey: str) -> SchedulingDataset:
    rows = fetch_rows_for_timekey(rule_timekey)
    if not rows:
        raise ValueError(f"No rows for RULE_TIMEKEY={rule_timekey}")
    return rows_to_dataset(rows, rule_timekey)


def save_conversions(records: list[ConversionRecord], rule_timekey: str) -> None:
    cfg = load_settings()["oracle"]
    out_table = cfg["output_table"]
    his_table = cfg.get("history_table")

    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute(f"DELETE FROM {out_table} WHERE RULE_TIMEKEY = :rtk", rtk=rule_timekey)
        insert_sql = f"""
            INSERT INTO {out_table} (
                RULE_TIMEKEY, FROM_BATCH, FROM_PLAN_PROD_KEY, FROM_OPER_ID,
                EQP_MODEL_CD, TO_BATCH_ID, TO_PLAN_PROD_KEY, TO_OPER_ID,
                START_CONV_TIME, EQP_QTY
            ) VALUES (
                :1,:2,:3,:4,:5,:6,:7,:8,:9,:10
            )
        """
        for rec in records:
            cur.execute(
                insert_sql,
                [
                    rec.rule_timekey,
                    rec.from_batch,
                    rec.from_plan_prod_key,
                    rec.from_oper_id,
                    rec.eqp_model_cd,
                    rec.to_batch_id,
                    rec.to_plan_prod_key,
                    rec.to_oper_id,
                    rec.start_conv_time,
                    rec.eqp_qty,
                ],
            )
        if his_table:
            cur.execute(
                f"INSERT INTO {his_table} SELECT * FROM {out_table} WHERE RULE_TIMEKEY = :rtk",
                rtk=rule_timekey,
            )
        conn.commit()
    finally:
        conn.close()
