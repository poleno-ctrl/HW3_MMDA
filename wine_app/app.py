import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics

from prediction import predict_batch, read_wines_csv


APP_DIR = Path(__file__).resolve().parent
st.set_page_config(page_title='Винная подборка', page_icon='🍷', layout='wide')


@st.cache_resource
def load_model(path: str, revision: int):
    return joblib.load(path)


@st.cache_data
def load_test(path: str, revision: int) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def test_proba(_model, X: pd.DataFrame, model_revision: int) -> np.ndarray:
    return _model.predict_proba(X)[:, 1]


try:
    model_path = APP_DIR / 'model.joblib'
    test_path = APP_DIR / 'test.csv'
    config = json.loads((APP_DIR / 'config.json').read_text())
    model_revision = model_path.stat().st_mtime_ns
    model = load_model(str(model_path), model_revision)
    test = load_test(str(test_path), test_path.stat().st_mtime_ns)
    features = list(model.feature_names_in_)
    X_test, y_test = test.loc[:, features], test['target'].to_numpy()
    scores = test_proba(model, X_test, model_revision)
except (OSError, ValueError, KeyError) as error:
    st.error(f'Не удалось загрузить сохранённую модель или данные: {error}')
    st.stop()

st.title('🍷 Винная подборка')
st.write('Оцените вино по лабораторному анализу и выберите, какие образцы включить в подборку.')
threshold = st.sidebar.slider(
    'Порог хорошего вина', min_value=0.0, max_value=1.0,
    value=float(config['threshold']), step=0.001, format='%.3f', key='threshold',
)
st.sidebar.caption('Один порог для всех вкладок. По умолчанию он выбран на валидации.')
st.sidebar.info(
    'Уменьшение порога расширяет подборку: хороших вин пропускаем меньше, '
    'но среди рекомендаций может стать больше посредственных. Увеличение '
    'порога сокращает подборку. Precision обычно растёт, но это не гарантия для каждого шага.'
)
single_tab, metrics_tab, batch_tab, coefficients_tab = st.tabs([
    'Одно вино', 'Модельные метрики', 'Пакетное предсказание', 'Что выучила модель',
])

labels = {
    'fixed_acidity': 'Фиксированная кислотность',
    'volatile_acidity': 'Летучая кислотность',
    'citric_acid': 'Лимонная кислота',
    'residual_sugar': 'Остаточный сахар',
    'chlorides': 'Хлориды',
    'free_sulfur_dioxide': 'Свободный диоксид серы',
    'total_sulfur_dioxide': 'Общий диоксид серы',
    'density': 'Плотность', 'pH': 'pH', 'sulphates': 'Сульфаты',
    'alcohol': 'Алкоголь',
}

with single_tab:
    st.subheader('Лабораторный анализ')
    st.caption('Диапазоны и начальные значения взяты из минимума, максимума и медианы тестовых данных.')
    columns = st.columns(3)
    values = {}
    for i, feature in enumerate(name for name in features if name != 'is_red'):
        series = X_test[feature]
        with columns[i % 3]:
            values[feature] = st.number_input(
                labels.get(feature, feature), min_value=float(series.min()),
                max_value=float(series.max()), value=float(series.median()),
                format='%.5f' if feature == 'density' else '%.3f', key=f'input_{feature}',
                help=feature,
            )
    values['is_red'] = int(st.toggle('Красное вино', value=bool(X_test['is_red'].median())))
    wine = pd.DataFrame([values]).loc[:, features]
    probability = float(model.predict_proba(wine)[0, 1])
    st.metric('Вероятность хорошего вина', f'{probability:.1%}')
    if probability >= threshold:
        st.success(f'Включить в подборку: вероятность ≥ {threshold:.3f}.')
    else:
        st.info(f'Пропустить: вероятность < {threshold:.3f}.')

with metrics_tab:
    st.subheader('Качество на отложенной выборке')
    st.caption(
        f'{len(y_test)} образцов; доля хороших вин — {y_test.mean():.1%}. '
        'Эта вкладка показывает поведение модели. Для выбора нового рабочего порога используйте валидацию.'
    )
    prediction = (scores >= threshold).astype(int)
    precision = metrics.precision_score(y_test, prediction, zero_division=0)
    recall = metrics.recall_score(y_test, prediction, zero_division=0)
    f1 = metrics.f1_score(y_test, prediction, zero_division=0)
    for column, name, value in zip(st.columns(3), ['Precision', 'Recall', 'F1'], [precision, recall, f1]):
        column.metric(name, f'{value:.3f}')
    st.write(f'Отобрано вин: **{prediction.sum()} из {len(prediction)}**.')
    if not prediction.any():
        st.info('При этом пороге ни одно вино не отобрано. Precision принят равным 0.')
    cm = metrics.confusion_matrix(y_test, prediction, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    left, middle, right = st.columns(3)
    fig, ax = plt.subplots(figsize=(4, 4))
    metrics.ConfusionMatrixDisplay(cm, display_labels=['Обычное', 'Хорошее']).plot(
        ax=ax, cmap='Blues', colorbar=False,
    )
    ax.set_title('Матрица ошибок')
    left.pyplot(fig)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(4, 4))
    metrics.RocCurveDisplay.from_predictions(y_test, scores, ax=ax, name='LR')
    ax.plot([0, 1], [0, 1], '--', color='grey')
    ax.scatter(fp / (fp + tn), recall, color='crimson', zorder=5, label=f'Порог {threshold:.3f}')
    ax.legend(fontsize=8)
    ax.set_title('ROC-кривая')
    middle.pyplot(fig)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(4, 4))
    metrics.PrecisionRecallDisplay.from_predictions(y_test, scores, ax=ax, name='LR')
    ax.axhline(y_test.mean(), linestyle='--', color='grey', label='Случайная модель')
    ax.scatter(recall, precision, color='crimson', zorder=5, label=f'Порог {threshold:.3f}')
    ax.legend(fontsize=8)
    ax.set_title('PR-кривая')
    right.pyplot(fig)
    plt.close(fig)
    st.caption('FP — рекомендовали посредственное вино; FN — пропустили хорошее.')

with batch_tab:
    st.subheader('Загрузите CSV с анализами вин')
    st.write(f'Порог отбора: **{threshold:.3f}**. Заголовки столбцов должны совпадать с примером.')
    st.download_button(
        'Скачать пример CSV', data=(APP_DIR / 'example_wines.csv').read_bytes(),
        file_name='example_wines.csv', mime='text/csv',
    )
    upload = st.file_uploader('CSV с 12 признаками, без target', type=['csv'])
    if upload is not None:
        try:
            uploaded_data = read_wines_csv(upload.getvalue())
            result = predict_batch(model, uploaded_data, features, threshold)
        except (ValueError, TypeError, UnicodeError, pd.errors.ParserError) as error:
            st.error(f'Не удалось обработать CSV: {error}')
        else:
            st.dataframe(result, hide_index=True)
            st.download_button(
                'Скачать прогнозы', result.to_csv(index=False).encode('utf-8-sig'),
                file_name='wine_predictions.csv', mime='text/csv',
            )

with coefficients_tab:
    st.subheader('Какие признаки связаны с оценкой')
    coefficients = pd.Series(model[-1].coef_.ravel(), index=features).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(coefficients.index, coefficients, color=np.where(coefficients >= 0, '#568c6b', '#b55d63'))
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set(xlabel='Коэффициент после StandardScaler', title='Влияние на логарифм шансов класса 1')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    positive = coefficients[coefficients > 0].tail(3).sort_values(ascending=False)
    negative = coefficients[coefficients < 0].head(3)
    st.write('Сильнее всего за хорошее вино: ' + ', '.join(positive.index) + '.')
    st.write('Сильнее всего против: ' + ', '.join(negative.index) + '.')
    st.caption(
        'Коэффициент описывает изменение логарифма шансов при росте признака на одно '
        'стандартное отклонение, если остальные признаки фиксированы. '
        'Это статистическая связь, а не доказательство причинного влияния.'
    )
