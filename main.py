import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose

# Заголовок приложения
st.title("Анализ продаж и прогнозирование")

# Боковая панель для загрузки файла
st.sidebar.header("Загрузите Excel файл")
uploaded_file = st.sidebar.file_uploader("Выберите файл Excel", type=["xlsx"])

if uploaded_file is not None:
    try:
        # Чтение данных
        data = pd.read_excel(uploaded_file)
        data['date'] = pd.to_datetime(data['date'])  # Преобразуем дату в формат datetime
        data['year_month'] = data['date'].dt.to_period('M')  # Создаем year_month

        st.write("### Загруженные данные:")
        st.write(data.head())

        # Проверяем наличие столбца remains
        if 'remains' in data.columns:
            # Используем remains как есть, без группировки, и сортируем по дате
            remains_data = data[['date', 'remains']].drop_duplicates(subset='date').sort_values(by='date')
        else:
            st.error("В загруженном файле отсутствует столбец 'remains'.")
            st.stop()

        # Сортируем данные по дате
        data = data.sort_values(by='date')

        # Агрегируем данные по месяцам
        monthly_data = data.groupby(data['year_month'])['sales'].sum()
        monthly_data.index = monthly_data.index.to_timestamp()  # Приводим индексы к Timestamp
        monthly_data = monthly_data.asfreq('MS')
        monthly_data = monthly_data.fillna(method='ffill')

        # Проверка длины данных для декомпозиции
        if len(monthly_data) < 12:
            st.error("Недостаточно данных для декомпозиции. Требуется минимум 12 месяцев данных.")
            st.stop()

        # Декомпозиция временного ряда
        decomposition = seasonal_decompose(monthly_data, model='multiplicative', period=12)

        # Извлечение компонентов декомпозиции
        seasonality = decomposition.seasonal
        trend = decomposition.trend

        # Преобразование в DataFrame для объединения
        seasonality_df = seasonality.reset_index()
        seasonality_df.columns = ['date', 'seasonality']

        trend_df = trend.reset_index()
        trend_df.columns = ['date', 'trend']

        # Убедимся, что индексы совпадают
        seasonality_df['date'] = pd.to_datetime(seasonality_df['date'])
        trend_df['date'] = pd.to_datetime(trend_df['date'])

        # Объединение с основными данными
        data = pd.merge(data, seasonality_df, on='date', how='left')
        data = pd.merge(data, trend_df, on='date', how='left')

        # Очистка от сезонности
        data['sales_deseasonalized'] = data['sales'] / data['seasonality']

        # Построение стандартного графика декомпозиции
        st.write("### Декомпозиция временного ряда")
        fig = decomposition.plot()
        fig.set_size_inches(14, 10)
        st.pyplot(fig)

        # Поля для ввода limit_up, limit_down, max_growth, max_decline
        st.sidebar.header("Параметры прогнозирования")
        limit_up = st.sidebar.number_input("Введите верхний лимит (%)", value=30, min_value=0, max_value=100, step=1)
        limit_down = st.sidebar.number_input("Введите нижний лимит (%)", value=-20, min_value=-100, max_value=0, step=1)
        max_growth = st.sidebar.number_input("Введите максимальный рост прогноза (%):", value=20, min_value=0,
                                              max_value=200, step=1)
        max_decline = st.sidebar.number_input("Введите максимальное падение прогноза (%):", value=-20, min_value=-100,
                                               max_value=0, step=1)

        calculation_type = st.sidebar.radio("Выберите тип расчёта для векторов:", options=["Максимум", "Среднее"])

        # Переключатель для применения сезонных коэффициентов
        apply_seasonality = st.sidebar.checkbox("Применять сезонные коэффициенты", value=True)

        # Сценарии
        scenarios = [
            {"name": "Годовой прогноз", "period_type": 'year'},
            {"name": "Месячный прогноз", "period_type": 'month'}
        ]

        # Агрегируем продажи по месяцам для прогнозирования
        monthly_sales = data.groupby(data['year_month'])['sales'].sum().reset_index()
        monthly_sales['date'] = monthly_sales['year_month'].dt.to_timestamp()

        results = {}

        for scenario in scenarios:
            forecast_results = []
            previous_forecast = None  # Переменная для хранения предыдущего прогноза
            for idx, row in monthly_sales.iterrows():
                month_date = row['date']

                if scenario['period_type'] == 'year':
                    fp12_date_start = month_date - pd.DateOffset(years=1)
                    fp18_date_start = month_date - pd.DateOffset(years=1, months=6)
                    fp18_date_end = month_date - pd.DateOffset(months=6)
                    fp24_date_start = month_date - pd.DateOffset(years=2)
                    fp36_date_start = month_date - pd.DateOffset(years=3)
                elif scenario['period_type'] == 'month':
                    fp12_date_start = month_date - pd.DateOffset(months=1)
                    fp18_date_start = month_date - pd.DateOffset(months=2)
                    fp18_date_end = month_date - pd.DateOffset(months=1)
                    fp24_date_start = month_date - pd.DateOffset(months=3)
                    fp36_date_start = month_date - pd.DateOffset(months=4)

                if len(data[(data['date'] >= fp36_date_start) & (data['date'] < month_date)]) < (
                        365 if scenario['period_type'] == 'year' else 30):
                    continue

                fp12 = data[(data['date'] >= fp12_date_start) & (data['date'] < month_date)]['sales'].sum()
                fp18 = data[(data['date'] >= fp18_date_start) & (data['date'] < fp18_date_end)]['sales'].sum()
                fp24 = data[(data['date'] >= fp24_date_start) & (data['date'] < fp12_date_start)]['sales'].sum()
                fp36 = data[(data['date'] >= fp36_date_start) & (data['date'] < fp24_date_start)]['sales'].sum()

                if fp12 == 0:
                    continue

                vector1 = (1 - fp18 / fp12) * 100
                vector2 = ((1 - fp24 / fp12) / 2) * 100
                vector3 = ((1 - fp36 / fp12) / 4) * 100

                # Логика расчёта vector_max
                if calculation_type == "Максимум":
                    vector_max = max(vector1, vector2, vector3)
                elif calculation_type == "Среднее":
                    vector_max = np.mean([vector1, vector2, vector3])

                # Применение лимитов, заданных пользователем
                if vector_max > 0:
                    vector_max_constrained = min(vector_max, limit_up)
                else:
                    vector_max_constrained = max(vector_max, limit_down)

                forecast = fp12 * (1 + vector_max_constrained / 100) / (
                    12 if scenario['period_type'] == 'year' else 1)

                # Применение сезонных коэффициентов
                if apply_seasonality:
                    forecast_with_seasonality = forecast * (1 + seasonality[month_date.month] - seasonality.mean())
                else:
                    forecast_with_seasonality = forecast

                # Применение ограничений на рост и падение прогноза
                if previous_forecast is not None:
                    max_allowed_forecast = previous_forecast * (1 + max_growth / 100)
                    min_allowed_forecast = previous_forecast * (1 + max_decline / 100)

                    if forecast_with_seasonality > max_allowed_forecast:
                        forecast_with_seasonality = max_allowed_forecast
                    elif forecast_with_seasonality < min_allowed_forecast:
                        forecast_with_seasonality = min_allowed_forecast

                previous_forecast = forecast_with_seasonality

                actual_sales = row['sales']
                forecast_results.append(
                    {"month_date": month_date, "forecast": forecast_with_seasonality, "actual_sales": actual_sales})

            results[scenario['name']] = pd.DataFrame(forecast_results)

        # Построение графика прогнозов и остатков
        plt.figure(figsize=(14, 7))

        # Фактические продажи
        plt.plot(monthly_sales['date'], monthly_sales['sales'], label='Фактические продажи (агрегированные)', color='black', linestyle='--')

        # Остатки
        plt.plot(remains_data['date'], remains_data['remains'], label='Остатки', color='blue', linestyle=':')

        # Прогнозы по сценариям
        for scenario in scenarios:
            scenario_name = scenario['name']
            df = results[scenario_name]

            if 'actual_sales' not in df.columns or 'forecast' not in df.columns:
                continue

            df = df.dropna(subset=['actual_sales', 'forecast'])
            mape = mean_absolute_percentage_error(df['actual_sales'], df['forecast']) * 100
            rmse = np.sqrt(mean_squared_error(df['actual_sales'], df['forecast']))

            plt.plot(df['month_date'], df['forecast'], label=f'{scenario_name} (MAPE: {mape:.2f}%, RMSE: {rmse:.2f})')

        # Оформление графика
        plt.title('Сравнение годового и месячного прогнозов с остатками')
        plt.xlabel('Дата')
        plt.ylabel('Продажи / Остатки')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=90)
        st.pyplot(plt)

    except Exception as e:
        st.error(f"Ошибка обработки файла: {e}")
else:
    st.info("Загрузите Excel файл для анализа.")
