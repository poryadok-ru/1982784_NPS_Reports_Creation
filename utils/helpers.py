import os
import calendar
import logging
from datetime import datetime, timedelta
from pathlib import Path
from log import Log
from config import PORADOCK_TOKEN


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


def delta(cur, prev) -> str:
    """В процентах изменение cur относительно prev с знаком."""
    try:
        # Оба аргумента числа и prev не ноль
        if isinstance(cur, (int, float)) and isinstance(prev, (int, float)):
            if prev == 0:
                return "деление на ноль"

            d = (cur - prev) / abs(prev) * 100
            return f"{d:+.1f}%"

        # Попытка преобразовать в float
        fc = float(cur) if cur not in (None, '') else float('nan')
        fp = float(prev) if prev not in (None, '') else float('nan')

        if fc != fc or fp != fp:  # Проверка на NaN
            raise ValueError("Нечисловые значения после преобразования")

        if fp == 0:
            return "деление на ноль"

        d = (fc - fp) / abs(fp) * 100
        return f"{d:+.1f}%"

    except (ValueError, TypeError):
        return "ошибка данных"
    except ZeroDivisionError:
        return "деление на ноль"
    except Exception:
        return "ошибка расчета"


def reviews_grouped_by_prk(df) -> dict:
    """Группировать отзывы по prk (uuid_code) -> list of dict."""
    grouped = df.groupby('prk')
    return {prk: reviews.to_dict(orient='records') for prk, reviews in grouped}


def setup_logger():
    """Настройка логгера."""
    if PORADOCK_TOKEN:
        logger_instance = Log(token=PORADOCK_TOKEN, silent_errors=True)
        return logger_instance
    else:
        # Используем стандартный logging если нет токена для Poradock
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('nps_reports.log', encoding='utf-8')
            ]
        )
        logger = logging.getLogger('nps_reports')
        return logger