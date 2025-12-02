from collections import Counter
from openai import OpenAI
from config import BASE_URL


def build_llm_analysis_prompt(
        store_name: str,
        current_data,
        prev_month_data,
        prev_year_data,
        statdict: dict
) -> str:
    """Построить промпт для LLM-анализа."""

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
ДАННЫЕ ДЛА АНАЛИЗА:
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


def analyze_via_litellm(prompt: str, model: str, api_key: str) -> str:
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