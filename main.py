import os
import json
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

from database.queries import (
    get_active_prks,
    get_all_prks_for_network_stats,
    fetch_nps_stats,
    fetch_reviews_data,
    fetch_llm_analysis_data_by_period,
    insert_report,
    get_llm_analysis_for_prk,
    get_prk_name_by_uuid
)
from services.llm_service import analyze_via_litellm, build_llm_analysis_prompt
from services.pdf_service import md_to_pdf
from clients.bitrix_client import upload_to_bitrix
from utils.helpers import (
    format_period_filename,
    prev_period,
    delta,
    reviews_grouped_by_prk,
    sanitize_filename,
    setup_logger
)
from config import (
    OUTPUT_DIR,
    MODEL_NAME,
    API_KEY
)


def get_period_based_on_today():
    """
    Определяет период отчета:
    - Если сегодня 1-е число: берет ПРЕДЫДУЩИЙ месяц полностью
    - Иначе: берет с 1-го числа ТЕКУЩЕГО месяца по вчерашний день
    """
    today = datetime.now()
    day_of_month = today.day

    if day_of_month == 1:
        # Если сегодня 1-е число, берем ПРЕДЫДУЩИЙ месяц полностью
        # Первый день текущего месяца
        first_day_of_current_month = today.replace(day=1)
        # Последний день предыдущего месяца
        last_day_of_prev_month = first_day_of_current_month - timedelta(days=1)
        # Первый день предыдущего месяца
        first_day_of_prev_month = last_day_of_prev_month.replace(day=1)

        period_start = first_day_of_prev_month.strftime('%Y-%m-%d')
        period_end = last_day_of_prev_month.strftime('%Y-%m-%d')

        return period_start, period_end
    else:
        # Если не 1-е число: с 1-го числа текущего месяца по вчерашний день
        period_start = today.replace(day=1).strftime('%Y-%m-%d')

        # Вчерашняя дата
        yesterday = today - timedelta(days=1)
        period_end = yesterday.strftime('%Y-%m-%d')

        return period_start, period_end


def main(output_dir: str = OUTPUT_DIR, manual_start: str = None, manual_end: str = None) -> None:
    """Главный запуск процесса генерации отчетов NPS."""
    logger = setup_logger()

    # Определяем период отчета
    if manual_start and manual_end:
        PERIOD_START, PERIOD_END = manual_start, manual_end
        logger.info(f"Используется ручной период: {PERIOD_START} - {PERIOD_END}")
    else:
        period = get_period_based_on_today()
        if period is None:
            logger.error("Не удалось определить период отчета")
            return
        PERIOD_START, PERIOD_END = period
        logger.info(f"Автоматический период: {PERIOD_START} - {PERIOD_END}")

    try:
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

        logger.info(f"Найдено {len(stores_now)} активных магазинов для анализа")

        output_all = {}
        processed_count, success_count, error_count = 0, 0, 0

        # 4. Отчеты по каждому магазину
        for prk_id, reviews_now in stores_now.items():
            if prk_id not in active_prks:
                continue
            processed_count += 1

            # Получаем название магазина по uuid_code
            store_name = get_prk_name_by_uuid(prk_id)

            # Создаем DataFrame из отзывов
            prk_df_now = pd.DataFrame(reviews_now)

            try:
                stats_now = fetch_nps_stats(prk_id, PERIOD_START, PERIOD_END, active_prks)
                stats_lm = fetch_nps_stats(prk_id, lm_start, lm_end, active_prks)
                stats_ly = fetch_nps_stats(prk_id, ly_start, ly_end, active_prks)
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

                # Фильтруем LLM анализ по ПРК
                llm_current_prk = get_llm_analysis_for_prk(llm_current, prk_id)
                llm_prev_month_prk = get_llm_analysis_for_prk(llm_prev_month, prk_id)
                llm_prev_year_prk = get_llm_analysis_for_prk(llm_prev_year, prk_id)
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
- **Текущий NPS магазина:** <span style="color:#123c3c">{statdict['nps']}</span> (разница с сетью: {delta(stats_now["nps"], network_now["nps"]) if isinstance(stats_now["nps"], (int, float)) and isinstance(network_now["nps"], (int, float)) else "н/д"})
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
                    detailed_analysis = analyze_via_litellm(analysis_prompt, MODEL_NAME, API_KEY)
                    report_md += "\n" + detailed_analysis
                except Exception as ex:
                    logger.error(f"Ошибка при ИИ анализе для {store_name}: {ex}")
                    report_md += f"\nОшибка при ИИ анализе для {store_name}: {ex}"
                    error_count += 1

                # Сохранение txt, pdf, Bitrix
                filename_base = f"NPS_отчет_{sanitize_filename(store_name)}_{period_str}"
                txt_path = Path(output_dir) / f"{filename_base}.txt"
                pdf_path = Path(output_dir) / f"{filename_base}.pdf"

                # Сохраняем TXT
                try:
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(report_md)
                except Exception as txt_ex:
                    logger.error(f"Ошибка при создании TXT для {store_name}: {txt_ex}")
                    error_count += 1
                    continue

                # Создаем PDF
                try:
                    md_to_pdf(report_md, str(pdf_path))
                except Exception as pdf_ex:
                    logger.error(f"Ошибка при создании PDF для {store_name}: {pdf_ex}")
                    error_count += 1
                    continue

                # Загружаем в Bitrix24 и сохраняем в БД
                download_url = None
                try:
                    download_url = upload_to_bitrix(str(pdf_path), store_name, period_str, logger)
                except Exception as bitrix_ex:
                    logger.error(f"Ошибка при загрузке в Bitrix24 для {store_name}: {bitrix_ex}")
                    error_count += 1

                # Сохраняем в БД, даже если не удалась загрузка в Bitrix24
                try:
                    if download_url:
                        insert_report(prk_id, download_url, PERIOD_START, PERIOD_END)
                        success_count += 1
                    else:
                        error_count += 1
                except Exception as db_ex:
                    logger.error(f"Ошибка при добавлении отчета {store_name} в reports: {db_ex}")
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

        # Финальная статистика
        logger.info(
            f"=== ОБРАБОТКА ЗАВЕРШЕНА ===\n"
            f"Период отчета: {PERIOD_START} - {PERIOD_END}\n"
            f"Всего активных ПРК: {len(active_prks)}\n"
            f"ПРК с отзывами за период: {len(stores_now)}\n"
            f"Успешно обработано: {success_count}\n"
            f"Ошибок: {error_count}"
        )
    except Exception as main_ex:
        logger.critical(f"Критическая ошибка в основном потоке: {main_ex}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Генерация отчетов NPS')
    parser.add_argument('--start', help='Начальная дата периода (YYYY-MM-DD)')
    parser.add_argument('--end', help='Конечная дата периода (YYYY-MM-DD)')
    parser.add_argument('--output', help='Директория для сохранения отчетов')

    args = parser.parse_args()

    # Запуск основной функции с аргументами
    main(
        output_dir=args.output or OUTPUT_DIR,
        manual_start=args.start,
        manual_end=args.end
    )