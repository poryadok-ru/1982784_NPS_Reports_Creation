import os
import json
import calendar
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter

import pandas as pd
import markdown
import pdfkit
import psycopg2
from openai import OpenAI
from dotenv import load_dotenv

from log import Log
from bitrix24_sdk import BitrixClient

# ====== Конфигурация и глобальные переменные ======
load_dotenv()

DB_PARAMS = {
    "host": os.getenv("NPS_DB_HOST"),
    "port": int(os.getenv("NPS_DB_PORT", "5432")),
    "dbname": os.getenv("NPS_DB_NAME"),
    "user": os.getenv("NPS_DB_USER"),
    "password": os.getenv("NPS_DB_PASSWORD"),
}

API_KEY = os.getenv("LLM_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
BASE_URL = os.getenv("LITELLM_BASE_URL")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
PERIOD_START = os.getenv("PERIOD_START")
PERIOD_END = os.getenv("PERIOD_END")
PORADOCK_TOKEN = os.getenv("PORADOCK_LOG_TOKEN")
BITRIX_NPS_REPORTS_FOLDER_ID = int(os.getenv("BITRIX_NPS_REPORTS_FOLDER_ID"))
BITRIX_TOKEN = os.getenv("BITRIX_TOKEN")
BITRIX_USER_ID = int(os.getenv("BITRIX_USER_ID", "0"))


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


def insert_or_update_report(prk: str, report_url: str) -> None:
    """Сохранить или обновить в БД ссылку на файл в Bitrix24."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO reports (prk, report)
                VALUES (%s, %s)
                ON CONFLICT (prk)
                DO UPDATE SET report = EXCLUDED.report, date = NOW()
                """,
                (prk, report_url)
            )
        conn.commit()


def fetch_reviews_data(start: str, end: str, prks_filter: dict | None = None) -> pd.DataFrame:
    """Загрузить отзывы за период с фильтрацией по ПРК."""
    with get_connection() as conn:
        prk_filter = ""
        if prks_filter:
            prk_list = "', '".join(prks_filter.keys())
            prk_filter = f"AND r.shop_guid IN ('{prk_list}')"
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
              {prk_filter}
              AND r.review_date BETWEEN '{start}' AND '{end}'
        """
        df = pd.read_sql(query, conn)
        return df


def fetch_llm_analysis_data_by_period(start: str, end: str, prks_filter: dict | None = None) -> pd.DataFrame:
    """Загрузить LLM-анализ отзывов за период с фильтрацией по ПРК."""
    with get_connection() as conn:
        prk_filter = ""
        if prks_filter:
            prk_list = "', '".join(prks_filter.keys())
            prk_filter = f"AND nr.shop_guid IN ('{prk_list}')"
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
              {prk_filter}
              AND nr.review_date BETWEEN '{start}' AND '{end}'
            GROUP BY nr.nps_comment, nr.order_id, nr.shop_guid, p.name
        """
        df = pd.read_sql(query, conn)
        return df


def get_llm_analysis_for_prk(llm_analysis_df: pd.DataFrame, prk: str) -> pd.DataFrame:
    """Отфильтровать LLM-анализ по ПРК."""
    if llm_analysis_df.empty:
        return pd.DataFrame()
    return llm_analysis_df[llm_analysis_df['shop_guid'] == prk]


def sanitize_filename(s: str) -> str:
    """Сделать строку безопасной для имени файла."""
    return "".join([c if c.isalnum() or c in ('-_') else '_' for c in str(s)])


def format_period_filename(start_date: str, end_date: str) -> str:
    """Форматировать период отчёта для имени файла."""
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    return f"{start.strftime('%Y-%m-%d')}_{end.strftime('%Y-%m-%d')}"


def minus_month(dt: datetime) -> datetime:
    """Получить дату на месяц назад (с учётом конца месяца)."""
    month = dt.month - 1 or 12
    year = dt.year - (1 if dt.month == 1 else 0)
    last_day = calendar.monthrange(year, month)[1]
    day = min(dt.day, last_day)
    return datetime(year, month, day)


def prev_period(period_start: str, period_end: str) -> tuple:
    """(start_lm, end_lm), (start_ly, end_ly) для сравнения с прошлым месяцем/годом."""
    start_dt = datetime.strptime(period_start, '%Y-%m-%d')
    end_dt = datetime.strptime(period_end, '%Y-%m-%d')
    delta = end_dt - start_dt
    prev_month_start = minus_month(start_dt)
    prev_month_end = prev_month_start + delta
    try:
        prev_year_start = start_dt.replace(year=start_dt.year - 1)
    except ValueError:
        prev_year_start = start_dt - timedelta(days=365)
    try:
        prev_year_end = end_dt.replace(year=end_dt.year - 1)
    except ValueError:
        prev_year_end = end_dt - timedelta(days=365)
    return (
        (prev_month_start.strftime('%Y-%m-%d'), prev_month_end.strftime('%Y-%m-%d')),
        (prev_year_start.strftime('%Y-%m-%d'), prev_year_end.strftime('%Y-%m-%d'))
    )


def fetch_nps_stats(prk_or_none: str | None, start: str, end: str, prks_filter: dict | None = None) -> dict:
    """Выгрузить статистику NPS."""
    with get_connection() as conn:
        where_prk = ""
        if prk_or_none:
            where_prk = f"AND r.shop_guid = '{prk_or_none}'"
        elif prks_filter:
            prk_list = "', '".join(prks_filter.keys())
            where_prk = f"AND r.shop_guid IN ('{prk_list}')"
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
            WHERE 1=1
              {where_prk}
              AND r.review_date BETWEEN '{start}' AND '{end}'
        """
        df = pd.read_sql(query, conn)
        row = df.iloc[0]
        return dict(
            nps=row["nps"] if pd.notnull(row["nps"]) else "-",
            cnt=int(row["total_cnt"]) if pd.notnull(row["total_cnt"]) else 0,
            promoters=int(row["promoters"]) if pd.notnull(row["promoters"]) else 0,
            passives=int(row["passives"]) if pd.notnull(row["passives"]) else 0,
            detractors=int(row["detractors"]) if pd.notnull(row["detractors"]) else 0,
        )


def delta(cur, prev) -> str:
    """В процентах изменение cur относительно prev с знаком."""
    if isinstance(cur, (int, float)) and isinstance(prev, (int, float)) and prev != 0:
        d = (cur - prev) / abs(prev) * 100
        return f"{d:+.1f}%"
    try:
        fc, fp = float(cur), float(prev)
        if fp != 0:
            d = (fc - fp) / abs(fp) * 100
            return f"{d:+.1f}%"
    except Exception:
        pass
    return "н/д"


def reviews_grouped_by_prk(df: pd.DataFrame) -> dict:
    """Группировать отзывы по prk (uuid_code) -> list of dict."""
    grouped = df.groupby('prk')
    return {prk: reviews.to_dict(orient='records') for prk, reviews in grouped}


def build_llm_analysis_prompt(
    store_name: str,
    current_data: pd.DataFrame,
    prev_month_data: pd.DataFrame,
    prev_year_data: pd.DataFrame,
    statdict: dict
) -> str:
    """Построить промпт для LLM-анализа (структура не изменена)."""
    # Внутренние вспомогательные функции не переписываются ради читабельности ответа
    def prepare_current_period_analysis(current_data):
        if current_data.empty:
            return "Нет данных за текущий период", "Нет примеров комментариев"
        negative_reasons = []
        for _, row in current_data.iterrows():
            if row['sentiments'] and any(sent == 'negative' for sent in row['sentiments'] if sent) and \
               row['reasons'] and any(row['reasons']):
                reasons = [reason for reason in row['reasons'] if reason and reason != 'None']
                negative_reasons.extend(reasons)
        if negative_reasons:
            reason_counts = Counter(negative_reasons)
            complaints_analysis = "\n".join([f"- {reason}" for reason, _ in reason_counts.most_common(8)])
        else:
            complaints_analysis = "Жалоб не обнаружено"
        comment_examples = [row['nps_comment'] for _, row in current_data.iterrows()
                            if row['nps_comment'] and row['nps_comment'].strip()]
        if comment_examples:
            unique_examples = list(set(comment_examples))[:5]
            examples_text = "\n".join([f"- '{example}'" for example in unique_examples])
        else:
            examples_text = "Нет примеров комментариев"
        return complaints_analysis, examples_text

    def prepare_comparative_analysis(current_data, prev_month_data, prev_year_data):
        analysis_points = []
        current_themes = []
        prev_month_themes = []
        prev_year_themes = []
        for _, row in current_data.iterrows():
            if row['theme_names']:
                current_themes.extend([theme for theme in row['theme_names'] if theme and theme != 'None'])
        for _, row in prev_month_data.iterrows():
            if row['theme_names']:
                prev_month_themes.extend([theme for theme in row['theme_names'] if theme and theme != 'None'])
        for _, row in prev_year_data.iterrows():
            if row['theme_names']:
                prev_year_themes.extend([theme for theme in row['theme_names'] if theme and theme != 'None'])
        current_sentiments = []
        prev_month_sentiments = []
        for _, row in current_data.iterrows():
            if row['sentiments']:
                current_sentiments.extend([sent for sent in row['sentiments'] if sent and sent != 'None'])
        for _, row in prev_month_data.iterrows():
            if row['sentiments']:
                prev_month_sentiments.extend([sent for sent in row['sentiments'] if sent and sent != 'None'])
        if prev_month_sentiments:
            current_negative = sum(1 for s in current_sentiments if s == 'negative')
            prev_negative = sum(1 for s in prev_month_sentiments if s == 'negative')
            if current_negative > prev_negative:
                analysis_points.append("рост числа негативных отзывов")
            elif current_negative < prev_negative:
                analysis_points.append("снижение числа негативных отзывов")
        if current_themes and prev_month_themes:
            current_theme_counts = Counter(current_themes)
            prev_theme_counts = Counter(prev_month_themes)
            for theme, count in current_theme_counts.most_common(5):
                prev_count = prev_theme_counts.get(theme, 0)
                if count > prev_count:
                    analysis_points.append(f"усиление проблем с {theme}")
                elif theme not in prev_theme_counts:
                    analysis_points.append(f"появление новых жалоб на {theme}")
        if prev_year_themes:
            prev_year_theme_counts = Counter(prev_year_themes)
            current_theme_counts = Counter(current_themes)
            for theme, _ in prev_year_theme_counts.most_common(3):
                if theme not in current_theme_counts:
                    analysis_points.append(f"исчезновение проблем с {theme} по сравнению с прошлым годом")
        return analysis_points

    complaints_analysis, examples_text = prepare_current_period_analysis(current_data)
    comparative_points = prepare_comparative_analysis(current_data, prev_month_data, prev_year_data)

    prompt = f"""
Ты — опытный аналитик в ритейле сети "Порядок". Тебе поручено провести анализ отзывов покупателей по магазину "{store_name}" и подготовить отчет для руководства.
СТАТИСТИКА NPS МАГАЗИНА:
Текущий NPS: {statdict['nps']}
NPS прошлого месяца: {statdict['nps_last_month']} ({statdict['nps_vs_lm']} к прошлому месяцу)
NPS прошлого года: {statdict['nps_last_year']} ({statdict['nps_vs_ly']} к прошлому году)
Количество отзывов: {statdict['cnt']}
Прошлый месяц: {statdict['cnt_last_month']} ({statdict['cnt_vs_lm']} к прошлому месяцу)
Прошлый год: {statdict['cnt_last_year']} ({statdict['cnt_vs_ly']} к прошлому году)
Промоутеры (9-10): {statdict['promoters']}
Нейтралы (7-8): {statdict['passives']}
Критики (0-6): {statdict['detractors']}
ДАННЫЕ ДЛЯ АНАЛИЗА:
Количество комментариев с LLM-анализом в текущем периоде: {len(current_data)}
ТВОИ ЗАДАЧИ:
1. ПРОАНАЛИЗИРОВАТЬ ЖАЛОБЫ за текущий период на основе поля "reason" из негативных отзывов (sentiment = negative)
2. ПРИВЕСТИ ПРИМЕРЫ комментариев, подтверждающих эти жалобы
3. ПРОВЕСТИ СРАВНИТЕЛЬНЫЙ АНАЛИЗ с предыдущими периодами
4. ВЫЯВИТЬ ВОЗМОЖНЫЕ ПРИЧИНЫ изменений
5. ДАТЬ КОНКРЕТНЫЕ РЕКОМЕНДАЦИИ
СФОРМАТИРУЙ ОТВЕТ СТРОГО ПО СЛЕДУЮЩЕЙ СТРУКТУРЕ:
## 2. Наиболее частые жалобы покупателей
{complaints_analysis}
## 3. Примеры комментариев
{examples_text}
## 4. Общая аналитика работы магазина
[Здесь проведи сравнительный анализ работы магазина. Опиши изменения по сравнению с прошлым месяцем и прошлым годом. 
Используй данные: {', '.join(comparative_points) if comparative_points else 'анализ изменений'}]
### Возможные причины изменений:
- [Выяви 2-3 основные возможные причины наблюдаемых изменений]
- [Сосредоточься на операционных и сервисных аспектах]
## 5. Рекомендации
- [Рекомендация 1 - конкретная и выполнимая]
- [Рекомендация 2 - конкретная и выполнимая] 
- [Рекомендация 3 - конкретная и выполнимая]
ВАЖНЫЕ ТРЕБОВАНИЯ:
- В разделах 2 и 3 говори только о текущем период (не упоминай прошлые периоды)
- В разделе 4 сделай акцент на сравнении и анализе причин
- Рекомендации должны быть практическими и конкретными
- Избегай общих фраз, опирайся на выявленные проблемы
- Не предлагай рекомендации, требующие глубокого знания бизнес-процессов
"""
    return prompt


def analyze_via_litellm(prompt: str, model: str = MODEL_NAME, api_key: str = API_KEY) -> str:
    """Получить анализ через LLM (OpenAI-совместимый endpoint)."""
    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Ты опытный аналитик данных ритейла. Ты анализируешь структурированные данные LLM-анализа отзывов и предоставляешь глубокую аналитику с конкретными рекомендациями."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=3500,
    )
    return response.choices[0].message.content.strip()


def md_to_pdf(md_text: str, pdf_path: str) -> None:
    """Сконвертировать markdown-отчет в PDF."""
    html_body = markdown.markdown(md_text, extensions=['fenced_code', 'tables'])
    html = f"""
    <html>
    <head>
    <meta charset="utf-8">
    <style>
        body {{ font-family: Arial, Verdana, Helvetica, sans-serif; margin: 2em; font-size: 14px; }}
        h1, h2, h3, h4 {{ color: #1a588b; }}
        code, pre {{ background: #f8f8f8; border-radius: 4px; padding: 5px 8px; font-size:13px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
        th, td {{ border: 1px solid #ccc; padding: 6px; }}
        th {{ background-color: #f0f0f5; }}
        ul, ol {{ margin-left: 1.5em; }}
        blockquote {{ border-left: 4px solid #ccc; margin: 1em; padding-left: 1em; color: #555; font-style: italic; background: #f9f9fd; }}
    </style>
    </head>
    <body>
    {html_body}
    </body>
    </html>
    """
    try:
        config = pdfkit.configuration()
    except Exception:
        config = pdfkit.configuration(wkhtmltopdf=r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe')
    pdfkit.from_string(html, pdf_path, configuration=config)


def upload_to_bitrix(pdf_path: str, store_name: str, period_str: str, logger) -> str:
    """Загрузить PDF отчет в Bitrix24 и вернуть публичную ссылку."""
    try:
        client = BitrixClient(token=BITRIX_TOKEN, user_id=BITRIX_USER_ID)
        folder_name = f"ПРК_{store_name}"
        target_folder_id = None

        nps_children = client.disk.get_children(id=BITRIX_NPS_REPORTS_FOLDER_ID)
        for item in nps_children.result:
            if item.name == folder_name and item.type == "folder":
                target_folder_id = item.id
                break
        if target_folder_id is None:
            new_folder = client.disk.add_subfolder(BITRIX_NPS_REPORTS_FOLDER_ID, {"NAME": folder_name})
            target_folder_id = new_folder.result.id
            logger.info(f"Создана новая папка для {store_name}: {folder_name} (ID: {target_folder_id})")
        with open(pdf_path, "rb") as f:
            file_content = f.read()
        file_name = f"NPS_отчет_{store_name}_{period_str}.pdf"
        upload_result = client.disk.upload_file_complete(
            folder_id=target_folder_id,
            file_content=file_content,
            file_name=file_name
        )
        download_url = getattr(
            upload_result.result,
            'download_url',
            getattr(
                upload_result.result, 'url',
                f"https://bitrix24.com/disk/downloadFile/{upload_result.result.id}/"
            )
        )
        return download_url
    except Exception as e:
        logger.error(f"Ошибка при загрузке в Bitrix24 для {store_name}: {e}")
        raise


def main(output_dir: str = OUTPUT_DIR) -> None:
    """Главный запуск процесса генерации отчетов NPS."""
    logger = setup_logger()
    try:
        logger.info("Запуск процесса генерации отчетов NPS")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        period_str = format_period_filename(PERIOD_START, PERIOD_END)

        # 1. Получение данных и периодов
        active_prks = get_active_prks()
        all_prks = get_all_prks_for_network_stats()
        if not active_prks:
            logger.error("Нет активных ПРК для анализа")
            return
        (lm_start, lm_end), (ly_start, ly_end) = prev_period(PERIOD_START, PERIOD_END)

        # 2. Сетевые NPS
        network_now = fetch_nps_stats(None, PERIOD_START, PERIOD_END, all_prks)
        network_lm = fetch_nps_stats(None, lm_start, lm_end, all_prks)
        network_ly = fetch_nps_stats(None, ly_start, ly_end, all_prks)

        # 3. LLM-анализ по ПРК
        llm_current = fetch_llm_analysis_data_by_period(PERIOD_START, PERIOD_END, active_prks)
        llm_prev_month = fetch_llm_analysis_data_by_period(lm_start, lm_end, active_prks)
        llm_prev_year = fetch_llm_analysis_data_by_period(ly_start, ly_end, active_prks)
        df_now = fetch_reviews_data(PERIOD_START, PERIOD_END, active_prks)
        stores_now = reviews_grouped_by_prk(df_now)

        logger.info(f"Найдено {len(stores_now)} активных магазинов с отзывами для анализа")

        output_all = {}
        processed_count, success_count, error_count = 0, 0, 0

        # 4. Отчеты по каждому магазину
        for prk, reviews_now in stores_now.items():
            if prk not in active_prks:
                logger.warning(f"ПРК {prk} больше не активен, пропускаем генерацию отчета")
                continue
            processed_count += 1
            prk_df_now = pd.DataFrame(reviews_now)
            store_name = prk_df_now['store_name'].iloc[0] if not prk_df_now.empty else "NO_NAME"
            try:
                stats_now = fetch_nps_stats(prk, PERIOD_START, PERIOD_END, active_prks)
                stats_lm = fetch_nps_stats(prk, lm_start, lm_end, active_prks)
                stats_ly = fetch_nps_stats(prk, ly_start, ly_end, active_prks)
                statdict = dict(
                    nps=stats_now['nps'],
                    nps_last_month=stats_lm['nps'],
                    nps_last_year=stats_ly['nps'],
                    nps_vs_lm=delta(stats_now['nps'], stats_lm['nps']),
                    nps_vs_ly=delta(stats_now['nps'], stats_ly['nps']),
                    cnt=stats_now['cnt'],
                    cnt_last_month=stats_lm['cnt'],
                    cnt_last_year=stats_ly['cnt'],
                    cnt_vs_lm=delta(stats_now['cnt'], stats_lm['cnt']),
                    cnt_vs_ly=delta(stats_now['cnt'], stats_ly['cnt']),
                    promoters=stats_now['promoters'],
                    passives=stats_now['passives'],
                    detractors=stats_now['detractors'],
                )
                llm_current_prk = get_llm_analysis_for_prk(llm_current, prk)
                llm_prev_month_prk = get_llm_analysis_for_prk(llm_prev_month, prk)
                llm_prev_year_prk = get_llm_analysis_for_prk(llm_prev_year, prk)
                period_text = f"{PERIOD_START} — {PERIOD_END}"

                report_md = f"""# Аналитика NPS магазина {store_name}
<p style="font-size:18px;"><b>Период отчёта: {period_text}</b></p>
## 1. Общая статистика NPS и динамика за период
### NPS по всей сети (все ПРК):
- **Текущий NPS по сети:** <span style="color:#284785">{network_now['nps']}</span>
- **NPS прошлого месяца:** <span style="color:#284785">{network_lm['nps']}</span> ({delta(network_now['nps'], network_lm['nps'])})
- **NPS прошлого года:** <span style="color:#284785">{network_ly['nps']}</span> ({delta(network_now['nps'], network_ly['nps'])})
- **Количество отзывов по сети:** <span style="color:#284785">{network_now['cnt']}</span>
- **Кол-во отзывов по сети в прошлом месяце:** <span style="color:#284785">{network_lm['cnt']}</span>
- **Кол-во отзывов по сети в прошлом году:** <span style="color:#284785">{network_ly['cnt']}</span>
### NPS данного ПРК:
- **Текущий NPS магазина:** <span style="color:#123c3c">{statdict['nps']}</span> (разница с сетью: {delta(stats_now["nps"], network_now["nps"]) if isinstance(stats_now["nps"], float) and isinstance(network_now["nps"], float) else "н/д"})
- **NPS прошлого месяца:** <span style="color:#123c3c">{statdict['nps_last_month']}</span> ({statdict['nps_vs_lm']} к прошлому месяцу)
- **NPS прошлого года:** <span style="color:#123c3c">{statdict['nps_last_year']}</span> ({statdict['nps_vs_ly']} к прошлому году)
- **Количество отзывов:** <span style="color:#123c3c">{statdict['cnt']}</span>
- **Прошлый месяц:** <span style="color:#123c3c">{statdict['cnt_last_month']}</span> ({statdict['cnt_vs_lm']} к прошлому месяцу)
- **Прошлый год:** <span style="color:#123c3c">{statdict['cnt_last_year']}</span> ({statdict['cnt_vs_ly']} к прошлому году)
- **Промоутеры (9-10):** <b>{statdict['promoters']}</b>
- **Нейтралы (7-8):** <b>{statdict['passives']}</b>
- **Критики (0-6):** <b>{statdict['detractors']}</b>
"""
                # LLM-анализ
                analysis_prompt = build_llm_analysis_prompt(
                    store_name, llm_current_prk, llm_prev_month_prk, llm_prev_year_prk, statdict)
                try:
                    detailed_analysis = analyze_via_litellm(analysis_prompt)
                    report_md += "\n" + detailed_analysis
                except Exception as ex:
                    logger.error(f"Ошибка при ИИ анализе для {store_name}: {ex}")
                    report_md += f"\nОшибка при ИИ анализе для {store_name}: {ex}"
                    error_count += 1

                # Сохранение txt, pdf, Bitrix
                filename_base = f"NPS_отчет_{sanitize_filename(store_name)}_{period_str}"
                txt_path = Path(output_dir) / f"{filename_base}.txt"
                pdf_path = Path(output_dir) / f"{filename_base}.pdf"
                try:
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(report_md)
                except Exception as txt_ex:
                    logger.error(f"Ошибка при создании TXT для {store_name}: {txt_ex}")
                    error_count += 1

                try:
                    md_to_pdf(report_md, str(pdf_path))
                    try:
                        download_url = upload_to_bitrix(str(pdf_path), store_name, period_str, logger)
                        try:
                            insert_or_update_report(store_name, download_url)
                            success_count += 1
                        except Exception as db_ex:
                            logger.error(f"Ошибка при добавлении отчета {store_name} в reports: {db_ex}")
                            error_count += 1
                    except Exception as bitrix_ex:
                        logger.error(f"Ошибка при загрузке в Bitrix24 для {store_name}: {bitrix_ex}")
                        error_count += 1
                except Exception as pdf_ex:
                    logger.error(f"Ошибка при создании PDF для {store_name}: {pdf_ex}")
                    error_count += 1

                output_all[store_name] = {
                    "final_report": report_md,
                    "statistics": statdict,
                    "network_now": network_now,
                    "llm_data_counts": {
                        "current": len(llm_current_prk),
                        "prev_month": len(llm_prev_month_prk),
                        "prev_year": len(llm_prev_year_prk)
                    }
                }
            except Exception as store_ex:
                error_msg = f"Критическая ошибка при обработке магазина {store_name}: {store_ex}"
                logger.error(error_msg)
                error_count += 1
                continue

        # 5. Итоговый JSON
        try:
            all_json_path = Path(output_dir) / f"all_analysis_{period_str}.json"
            with open(all_json_path, 'w', encoding='utf-8') as f:
                json.dump(output_all, f, ensure_ascii=False, indent=2)
        except Exception as json_ex:
            logger.error(f"Ошибка при сохранении сводного файла: {json_ex}")

        logger.info(
            f"=== ОБРАБОТКА ЗАВЕРШЕНА ===\n"
            f"Всего активных ПРК: {len(active_prks)}\n"
            f"ПРК с отзывами за период: {len(stores_now)}\n"
            f"Успешно обработано: {success_count}\n"
            f"Ошибок: {error_count}"
        )
    except Exception as main_ex:
        logger.critical(f"Критическая ошибка в основном потоке: {main_ex}")
        raise


def setup_logger():
    """Конструктор логгера с поддержкой заглушки."""
    if PORADOCK_TOKEN:
        logger_instance = Log(token=PORADOCK_TOKEN, silent_errors=True)
        return logger_instance
    else:
        print("Предупреждение: PORADOCK_LOG_TOKEN не установлен. Логирование отключено.")

        class DummyLogger:
            def __enter__(self): return self
            def __exit__(self, exc_type, exc_val, exc_tb): return False
            def info(self, msg): print(f"INFO: {msg}")
            def error(self, msg): print(f"ERROR: {msg}")
            def critical(self, msg): print(f"CRITICAL: {msg}")
            def warning(self, msg): print(f"WARNING: {msg}")
            def finish_success(self, *a, **k): print("INFO: finish_success called")
            def finish_error(self, *a, **k): print("ERROR: finish_error called")
        return DummyLogger()

if __name__ == "__main__":
    main()