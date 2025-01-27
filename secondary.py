import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose

st.set_page_config(
    page_title="Анализ продаж и прогнозирование",  # Название страницы
    layout="wide",  # Широкий режим
    initial_sidebar_state="expanded"  # Оставить боковую панель открытой
)

# Заголовок приложения
st.title("Анализ продаж, прогнозирование и остатки на складе")

# Боковая панель для загрузки файла
st.sidebar.header("Загрузите Excel файл")
uploaded_file = st.sidebar.file_uploader("Выберите файл Excel", type=["xlsx"])

if uploaded_file is not None:
    try:
        # Чтение данных
        data = pd.read_excel(uploaded_file)
        data['date'] = pd.to_datetime(data['date'])
        data['year_month'] = data['date'].dt.to_period('M')

        st.write("### Загруженные данные:")
        st.write(data.head())

        # Проверяем наличие столбца remains
        if 'remains' in data.columns:
            remains_data = data[['date', 'remains']].drop_duplicates(subset='date').sort_values(by='date')
        else:
            st.error("В загруженном файле отсутствует столбец 'remains'.")
            st.stop()

        # Сортируем данные по дате
        data = data.sort_values(by='date')

        # Агрегируем данные по месяцам
        monthly_data = data.groupby(data['year_month'])['sales'].sum()
        monthly_data.index = monthly_data.index.to_timestamp()
        monthly_data = monthly_data.asfreq('MS')
        monthly_data = monthly_data.fillna(method='ffill')

        # Декомпозиция временного ряда
        decomposition = seasonal_decompose(monthly_data, model='multiplicative', period=12)
        seasonality = decomposition.seasonal
        trend = decomposition.trend

        # Поля для ввода параметров прогноза
        st.sidebar.header("Параметры прогноза")
        limit_up = st.sidebar.number_input("Введите верхний лимит (%)", value=30, min_value=0, max_value=100, step=1)
        limit_down = st.sidebar.number_input("Введите нижний лимит (%)", value=-20, min_value=-100, max_value=0, step=1)
        calculation_type = st.sidebar.radio("Выберите тип расчёта для векторов:", options=["Максимум", "Среднее"])

        # Чекбокс для включения/отключения сезонности
        apply_seasonality = st.sidebar.checkbox("Применять сезонные коэффициенты", value=True)

        # Поля для задания временных интервалов fp12, fp18, fp24, fp36
        st.sidebar.header("Параметры временных интервалов (в месяцах)")
        fp12_months = st.sidebar.number_input("Горизонт анализа (в месяцах):", value=12, min_value=1, max_value=60, step=1)
        fp18_months = st.sidebar.number_input("Первый вектор (в месяцах):", value=18, min_value=1, max_value=60, step=1)
        fp24_months = st.sidebar.number_input("Второй вектор (в месяцах) / 2:", value=24, min_value=1, max_value=60, step=1)
        fp36_months = st.sidebar.number_input("Третий вектор (в месяцах) / 4:", value=36, min_value=1, max_value=120, step=1)

        # Поля для ввода ограничений на рост и снижение прогноза
        st.sidebar.header("Ограничения на прогноз")
        max_growth = st.sidebar.number_input("Максимальный рост прогноза (%)", value=20, min_value=0, max_value=200, step=1)
        max_decline = st.sidebar.number_input("Максимальное снижение прогноза (%)", value=-20, min_value=-100, max_value=0, step=1)

        # Рассчитываем прогноз
        monthly_sales = data.groupby(data['year_month'])['sales'].sum().reset_index()
        monthly_sales['date'] = monthly_sales['year_month'].dt.to_timestamp()

        results = []
        previous_forecast = None
        for idx, row in monthly_sales.iterrows():
            month_date = row['date']

            # Используем пользовательские интервалы
            fp12_date_start = month_date - pd.DateOffset(months=fp12_months)
            fp18_date_start = month_date - pd.DateOffset(months=fp18_months)
            fp18_date_end = month_date - pd.DateOffset(months=fp12_months)
            fp24_date_start = month_date - pd.DateOffset(months=fp24_months)
            fp36_date_start = month_date - pd.DateOffset(months=fp36_months)

            fp12 = data[(data['date'] >= fp12_date_start) & (data['date'] < month_date)]['sales'].sum()
            fp18 = data[(data['date'] >= fp18_date_start) & (data['date'] < fp18_date_end)]['sales'].sum()
            fp24 = data[(data['date'] >= fp24_date_start) & (data['date'] < fp12_date_start)]['sales'].sum()
            fp36 = data[(data['date'] >= fp36_date_start) & (data['date'] < fp24_date_start)]['sales'].sum()

            if fp12 == 0:
                continue

            vector1 = (1 - fp18 / fp12) * 100
            vector2 = ((1 - fp24 / fp12) / 2) * 100
            vector3 = ((1 - fp36 / fp12) / 4) * 100

            if calculation_type == "Максимум":
                vector_max = max(vector1, vector2, vector3)
            else:
                vector_max = np.mean([vector1, vector2, vector3])

            if vector_max > 0:
                vector_max_constrained = min(vector_max, limit_up)
            else:
                vector_max_constrained = max(vector_max, limit_down)

            average_sales_per_month = fp12 / fp12_months
            forecast = average_sales_per_month * (1 + vector_max_constrained / 100)

            # Применение сезонности (если включено)
            if apply_seasonality:
                forecast_with_seasonality = forecast * (1 + seasonality[month_date.month] - seasonality.mean())
            else:
                forecast_with_seasonality = forecast

            # Применение ограничений на рост и снижение прогноза
            if previous_forecast is not None:
                max_allowed_forecast = previous_forecast * (1 + max_growth / 100)
                min_allowed_forecast = previous_forecast * (1 + max_decline / 100)

                if forecast_with_seasonality > max_allowed_forecast:
                    forecast_with_seasonality = max_allowed_forecast
                elif forecast_with_seasonality < min_allowed_forecast:
                    forecast_with_seasonality = min_allowed_forecast

            previous_forecast = forecast_with_seasonality

            actual_sales = row['sales']
            results.append({"month_date": month_date, "forecast": forecast_with_seasonality, "actual_sales": actual_sales})

        results_df = pd.DataFrame(results)

        # Проверка, есть ли данные для расчёта метрик
        if not results_df.empty:
            # Убираем строки с отсутствующими фактическими данными
            results_df = results_df.dropna(subset=['actual_sales', 'forecast'])

            # Метрики
            mae = np.mean(np.abs(results_df['actual_sales'] - results_df['forecast']))
            rmse = np.sqrt(np.mean((results_df['actual_sales'] - results_df['forecast']) ** 2))
            mape = np.mean(
                np.abs((results_df['actual_sales'] - results_df['forecast']) / results_df['actual_sales'])) * 100

            # Вывод метрик
            st.write("### Метрики прогноза:")
            st.write(f"- **MAE (средняя абсолютная ошибка):** {mae:.2f}")
            st.write(f"- **RMSE (среднеквадратичная ошибка):** {rmse:.2f}")
            st.write(f"- **MAPE (средняя абсолютная процентная ошибка):** {mape:.2f}%")
        else:
            st.warning("Недостаточно данных для расчёта метрик.")

        # График прогнозов и остатков
        st.write("### Прогноз и остатки на складе")
        plt.figure(figsize=(14, 7))

        # Фактические продажи
        plt.plot(monthly_sales['date'], monthly_sales['sales'], label='Фактические продажи', color='black', linestyle='--')

        # Прогноз
        if not results_df.empty:
            plt.plot(results_df['month_date'], results_df['forecast'], label='Прогноз', color='green')

        # Остатки
        plt.plot(remains_data['date'], remains_data['remains'], label='Остатки', color='blue', linestyle=':')

        plt.title("Прогноз и остатки на складе")
        plt.xlabel('Дата')
        plt.ylabel('Продажи / Остатки')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=90)
        st.pyplot(plt)

        # График сезонности и тренда
        st.write("### График сезонности и тренда")
        fig = decomposition.plot()
        fig.set_size_inches(14, 10)
        st.pyplot(fig)

        # Проверка, достаточно ли данных для декомпозиции
        if len(monthly_data) < 12:
            st.error("Недостаточно данных для декомпозиции. Требуется минимум 12 точек.")
            st.stop()

        # Декомпозиция временного ряда
        decomposition = seasonal_decompose(monthly_data, model='multiplicative', period=12)
        trend = decomposition.trend
        seasonality = decomposition.seasonal

        # Удаляем строки с NaN из тренда и сезонности
        trend_cleaned = trend.dropna()
        seasonality_cleaned = seasonality.dropna()

        # Проверяем, что их индексы совпадают
        common_index = trend_cleaned.index.intersection(seasonality_cleaned.index)

        # Синхронизируем тренд и сезонность по общим индексам
        trend_cleaned = trend_cleaned.loc[common_index]
        seasonality_cleaned = seasonality_cleaned.loc[common_index]

        # Если необходимо объединить тренд и сезонность в единый DataFrame:
        trend_seasonal_combined = pd.DataFrame({
            'trend': trend_cleaned,
            'seasonality': seasonality_cleaned
        })

        # Проверяем, что тренд и сезонность существуют
        if trend_cleaned.isnull().any() or seasonality_cleaned.isnull().any():


            st.warning("Недостаточно данных для построения прогноза на основе тренда с сезонностью.")
        else:
            # Рассчитываем прогноз на основе тренда и сезонности
            trend_seasonal_forecast = trend * seasonality
            trend_seasonal_forecast = trend_seasonal_forecast.dropna()  # Убираем пропущенные значения

            # Сравниваем прогноз с фактическими данными
            common_index = trend_seasonal_forecast.index.intersection(monthly_data.index)
            actual_sales = monthly_data.loc[common_index]
            forecast = trend_seasonal_forecast.loc[common_index]

            # Вычисляем метрики
            mae = np.mean(np.abs(actual_sales - forecast))
            rmse = np.sqrt(np.mean((actual_sales - forecast) ** 2))
            mape = np.mean(np.abs((actual_sales - forecast) / actual_sales)) * 100


            # Рассчитываем прогноз как произведение тренда и сезонности
            trend_seasonal_forecast = trend_cleaned * seasonality_cleaned
            trend_seasonal_forecast = trend_seasonal_forecast.dropna()  # Убираем пропущенные значения

            # # График
            # plt.figure(figsize=(14, 7))
            #
            # # Прогноз на основе тренда и сезонности
            # plt.plot(trend_seasonal_forecast.index, trend_seasonal_forecast, label='Прогноз (тренд + сезонность)',
            #          color='green')
            #
            # # Фактические продажи
            # plt.plot(monthly_data.index, monthly_data, label='Фактические продажи', color='black', linestyle='--')
            #
            # # Оформление графика
            # plt.title("Прогноз на основе тренда с учётом сезонности")
            # plt.xlabel("Дата")
            # plt.ylabel("Продажи")
            # plt.legend()
            # plt.grid(True)
            # plt.xticks(rotation=90)
            # st.pyplot(plt)

            # Проверяем наличие данных тренда и сезонности
            if not trend_cleaned.isnull().any() and not seasonality_cleaned.isnull().any():
                # Рассчитываем прогноз на основе тренда и сезонности
                trend_seasonal_forecast = trend_cleaned * seasonality_cleaned
                trend_seasonal_forecast = trend_seasonal_forecast.dropna()  # Убираем пропущенные значения

                # Сравниваем прогноз с фактическими данными
                common_index = trend_seasonal_forecast.index.intersection(monthly_data.index)
                actual_sales = monthly_data.loc[common_index]
                forecast = trend_seasonal_forecast.loc[common_index]

                # Экстраполяция тренда
                last_trend_value = trend.dropna().iloc[-1]
                trend_slope = trend.diff().mean()  # Средний наклон тренда
                future_trend = [last_trend_value + (i + 1) * trend_slope for i in range(12)]

                # Экстраполяция сезонности (повторяем первый год)
                future_seasonality = seasonality.iloc[:12].values

                # Прогноз на 12 месяцев вперёд
                future_forecast = [t * s for t, s in zip(future_trend, future_seasonality)]

                # Даты для будущих прогнозов
                last_date = trend.dropna().index[-1]
                future_dates = [last_date + pd.DateOffset(months=i + 1) for i in range(12)]

                # Создание DataFrame для будущего прогноза
                future_forecast_df = pd.DataFrame({
                    'date': future_dates,
                    'forecast': future_forecast
                })

                # Вывод метрик (MAE, RMSE, MAPE) для исторического прогноза
                mae = np.mean(np.abs(actual_sales - forecast))
                rmse = np.sqrt(np.mean((actual_sales - forecast) ** 2))
                mape = np.mean(np.abs((actual_sales - forecast) / actual_sales)) * 100

                # Построение графика
                plt.figure(figsize=(14, 7))

                # Фактические продажи
                plt.plot(monthly_data.index, monthly_data, label='Фактические продажи', color='black', linestyle='--')

                # Исторический прогноз на основе тренда и сезонности
                plt.plot(forecast.index, forecast, label='Прогноз (исторический)', color='green')

                # Прогноз на будущее
                plt.plot(future_forecast_df['date'], future_forecast_df['forecast'], label='Прогноз (на будущее)',
                         color='blue')

                # Вывод метрик на график
                plt.text(0.02, 0.95, f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nMAPE: {mape:.2f}%",
                         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                         bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

                # Оформление графика
                plt.title("Прогноз на основе тренда и сезонности с фактическими продажами")
                plt.xlabel("Дата")
                plt.ylabel("Продажи")
                plt.legend()
                plt.grid(True)
                plt.xticks(rotation=90)
                st.pyplot(plt)
            else:
                st.warning("Недостаточно данных для расчёта прогнозов на основе тренда с сезонностью.")
    except Exception as e:
        st.error(f"Ошибка обработки файла: {e}")
else:
    st.info("Загрузите Excel файл для анализа.")
