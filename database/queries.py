import pandas as pd
import psycopg2
from config import DB_PARAMS


def get_connection():
    """Создать соединение с БД."""
    return psycopg2.connect(**DB_PARAMS)


def get_active_prks() -> dict:
    """Вернуть dict с uuid_code:name активных ПРК."""
    with get_connection() as conn:
        query = "SELECT uuid_code, name FROM prk WHERE is_active = true"
        df = pd.read_sql(query, conn)
        return dict(zip(df['uuid_code'], df['name']))


def get_all_prks_for_network_stats() -> dict:
    """Вернуть dict всех ПРК (активных и нет) uuid_code:name."""
    with get_connection() as conn:
        query = "SELECT uuid_code, name FROM prk"
        df = pd.read_sql(query, conn)
        return dict(zip(df['uuid_code'], df['name']))


def insert_report(prk_id: str, report_url: str, period_from: str, period_to: str) -> None:
    """Вставить новую запись отчета в БД."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO reports (prk_id, report, period_from, period_to, date)
                VALUES (%s, %s, %s, %s, NOW())
                """,
                (prk_id, report_url, period_from, period_to)
            )
        conn.commit()


def get_prk_name_by_uuid(uuid_code: str) -> str:
    """Получить название ПРК по uuid_code."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT name FROM prk WHERE uuid_code = %s",
                (uuid_code,)
            )
            result = cur.fetchone()
            return result[0] if result else "UNKNOWN_PRK"


def fetch_reviews_data(start: str, end: str, prks_filter: dict | None = None) -> pd.DataFrame:
    """Загрузить отзывы за период с фильтрацией по ПРК."""
    with get_connection() as conn:
        params = [start, end]

        if prks_filter:
            prk_placeholders = ','.join(['%s'] * len(prks_filter))
            query = f"""
                SELECT 
                    r.shop_guid AS prk,
                    r.nps_rating,
                    r.nps_comment,
                    r.nps_drawbacks,
                    r.review_date,
                    r.channel,
                    p.name AS store_name,
                    p.city,
                    p.region
                FROM nps_reviews r
                LEFT JOIN prk p ON r.shop_guid = p.uuid_code
                WHERE 1=1
                  AND r.shop_guid IN ({prk_placeholders})
                  AND r.review_date BETWEEN %s AND %s
            """
            params = list(prks_filter.keys()) + params
        else:
            query = """
                SELECT 
                    r.shop_guid AS prk,
                    r.nps_rating,
                    r.nps_comment,
                    r.nps_drawbacks,
                    r.review_date,
                    r.channel,
                    p.name AS store_name,
                    p.city,
                    p.region
                FROM nps_reviews r
                LEFT JOIN prk p ON r.shop_guid = p.uuid_code
                WHERE r.review_date BETWEEN %s AND %s
            """

        df = pd.read_sql(query, conn, params=params)
        return df


def fetch_llm_analysis_data_by_period(start: str, end: str, prks_filter: dict | None = None) -> pd.DataFrame:
    """Загрузить LLM-анализ отзывов за период с фильтрацией по ПРК."""
    with get_connection() as conn:
        params = [start, end]

        if prks_filter:
            prk_placeholders = ','.join(['%s'] * len(prks_filter))
            query = f"""
                SELECT 
                    nr.nps_comment,
                    nr.order_id,
                    nr.shop_guid,
                    p.name as store_name,
                    array_agg(DISTINCT t.name) as theme_names,
                    array_agg(DISTINCT st.name) as subtheme_names,
                    array_agg(DISTINCT lca.intensity) as intensities,
                    array_agg(DISTINCT lca.sentiment) as sentiments,
                    array_agg(DISTINCT lca.reason) as reasons
                FROM nps_reviews nr
                LEFT JOIN llm_comment_analysis lca ON nr.order_id = lca.order_id
                LEFT JOIN themes t ON lca.theme_id = t.id
                LEFT JOIN subthemes st ON lca.subtheme_id = st.id
                LEFT JOIN prk p ON nr.shop_guid = p.uuid_code
                WHERE nr.nps_comment IS NOT NULL 
                  AND nr.nps_comment != ''
                  AND TRIM(nr.nps_comment) != ''
                  AND LENGTH(TRIM(nr.nps_comment)) > 0
                  AND nr.shop_guid IN ({prk_placeholders})
                  AND nr.review_date BETWEEN %s AND %s
                GROUP BY nr.nps_comment, nr.order_id, nr.shop_guid, p.name
            """
            params = list(prks_filter.keys()) + params
        else:
            query = """
                SELECT 
                    nr.nps_comment,
                    nr.order_id,
                    nr.shop_guid,
                    p.name as store_name,
                    array_agg(DISTINCT t.name) as theme_names,
                    array_agg(DISTINCT st.name) as subtheme_names,
                    array_agg(DISTINCT lca.intensity) as intensities,
                    array_agg(DISTINCT lca.sentiment) as sentiments,
                    array_agg(DISTINCT lca.reason) as reasons
                FROM nps_reviews nr
                LEFT JOIN llm_comment_analysis lca ON nr.order_id = lca.order_id
                LEFT JOIN themes t ON lca.theme_id = t.id
                LEFT JOIN subthemes st ON lca.subtheme_id = st.id
                LEFT JOIN prk p ON nr.shop_guid = p.uuid_code
                WHERE nr.nps_comment IS NOT NULL 
                  AND nr.nps_comment != ''
                  AND TRIM(nr.nps_comment) != ''
                  AND LENGTH(TRIM(nr.nps_comment)) > 0
                  AND nr.review_date BETWEEN %s AND %s
                GROUP BY nr.nps_comment, nr.order_id, nr.shop_guid, p.name
            """

        df = pd.read_sql(query, conn, params=params)
        return df


def get_llm_analysis_for_prk(llm_analysis_df: pd.DataFrame, prk: str) -> pd.DataFrame:
    """Отфильтровать LLM-анализ по ПРК."""
    if llm_analysis_df.empty:
        return pd.DataFrame()
    return llm_analysis_df[llm_analysis_df['shop_guid'] == prk]


def fetch_nps_stats(prk_or_none: str | None, start: str, end: str, prks_filter: dict | None = None) -> dict:
    """Выгрузить статистику NPS."""
    with get_connection() as conn:
        params = [start, end]

        if prk_or_none:
            query = """
                SELECT
                  COUNT(*) AS total_cnt,
                  SUM(CASE WHEN nps_rating BETWEEN 9 AND 10 THEN 1 ELSE 0 END) AS promoters,
                  SUM(CASE WHEN nps_rating BETWEEN 7 AND 8 THEN 1 ELSE 0 END) AS passives,
                  SUM(CASE WHEN nps_rating BETWEEN 0 AND 6 THEN 1 ELSE 0 END) AS detractors,
                  CASE WHEN COUNT(*)>0 THEN ROUND(
                    100.0 * (SUM(CASE WHEN nps_rating BETWEEN 9 AND 10 THEN 1 ELSE 0 END)
                    - SUM(CASE WHEN nps_rating BETWEEN 0 AND 6 THEN 1 ELSE 0 END)) / COUNT(*), 1)
                  ELSE NULL END AS nps
                FROM nps_reviews r
                LEFT JOIN prk p ON r.shop_guid = p.uuid_code
                WHERE r.shop_guid = %s
                  AND r.review_date BETWEEN %s AND %s
            """
            params = [prk_or_none] + params
        elif prks_filter:
            prk_placeholders = ','.join(['%s'] * len(prks_filter))
            query = f"""
                SELECT
                  COUNT(*) AS total_cnt,
                  SUM(CASE WHEN nps_rating BETWEEN 9 AND 10 THEN 1 ELSE 0 END) AS promoters,
                  SUM(CASE WHEN nps_rating BETWEEN 7 AND 8 THEN 1 ELSE 0 END) AS passives,
                  SUM(CASE WHEN nps_rating BETWEEN 0 AND 6 THEN 1 ELSE 0 END) AS detractors,
                  CASE WHEN COUNT(*)>0 THEN ROUND(
                    100.0 * (SUM(CASE WHEN nps_rating BETWEEN 9 AND 10 THEN 1 ELSE 0 END)
                    - SUM(CASE WHEN nps_rating BETWEEN 0 AND 6 THEN 1 ELSE 0 END)) / COUNT(*), 1)
                  ELSE NULL END AS nps
                FROM nps_reviews r
                LEFT JOIN prk p ON r.shop_guid = p.uuid_code
                WHERE r.shop_guid IN ({prk_placeholders})
                  AND r.review_date BETWEEN %s AND %s
            """
            params = list(prks_filter.keys()) + params
        else:
            query = """
                SELECT
                  COUNT(*) AS total_cnt,
                  SUM(CASE WHEN nps_rating BETWEEN 9 AND 10 THEN 1 ELSE 0 END) AS promoters,
                  SUM(CASE WHEN nps_rating BETWEEN 7 AND 8 THEN 1 ELSE 0 END) AS passives,
                  SUM(CASE WHEN nps_rating BETWEEN 0 AND 6 THEN 1 ELSE 0 END) AS detractors,
                  CASE WHEN COUNT(*)>0 THEN ROUND(
                    100.0 * (SUM(CASE WHEN nps_rating BETWEEN 9 AND 10 THEN 1 ELSE 0 END)
                    - SUM(CASE WHEN nps_rating BETWEEN 0 AND 6 THEN 1 ELSE 0 END)) / COUNT(*), 1)
                  ELSE NULL END AS nps
                FROM nps_reviews r
                LEFT JOIN prk p ON r.shop_guid = p.uuid_code
                WHERE r.review_date BETWEEN %s AND %s
            """

        df = pd.read_sql(query, conn, params=params)
        row = df.iloc[0]
        return dict(
            nps=row["nps"] if pd.notnull(row["nps"]) else "-",
            cnt=int(row["total_cnt"]) if pd.notnull(row["total_cnt"]) else 0,
            promoters=int(row["promoters"]) if pd.notnull(row["promoters"]) else 0,
            passives=int(row["passives"]) if pd.notnull(row["passives"]) else 0,
            detractors=int(row["detractors"]) if pd.notnull(row["detractors"]) else 0,
        )