import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics


APP_DIR = Path(__file__).resolve().parent
st.set_page_config(page_title='Винная подборка', layout='wide')


@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / 'model.joblib')


@st.cache_data
def load_test():
    return pd.read_csv(APP_DIR / 'test.csv')


@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


model = load_model()
test = load_test()
config = json.loads((APP_DIR / 'config.json').read_text())
features = list(model.feature_names_in_)
X_test = test[features]
y_test = test['target'].to_numpy()
scores = test_proba(model, X_test)

st.title('Винная подборка')
threshold = st.sidebar.slider(
    'Порог хорошего вина', min_value=0.0, max_value=1.0,
    value=float(config['threshold']), step=0.001, format='%.3f', key='threshold',
)
st.sidebar.write(
    'При повышении порога подборка становится меньше, обычно в ней меньше посредственных вин. '
    'При снижении порога подборка расширяется, но в нее может попасть больше посредственных вин.'
)
single_tab, metrics_tab, batch_tab, coefficients_tab = st.tabs([
    'Одно вино', 'Модельные метрики', 'Пакетное предсказание', 'Что выучила модель',
])

with single_tab:
    columns = st.columns(3)
    values = {}
    for i, feature in enumerate(name for name in features if name != 'is_red'):
        series = X_test[feature]
        with columns[i % 3]:
            values[feature] = st.number_input(
                feature, min_value=float(series.min()), max_value=float(series.max()),
                value=float(series.median()), step=0.00001 if feature == 'density' else 0.01,
                format='%.5f' if feature == 'density' else '%.3f',
            )
    values['is_red'] = int(st.toggle('Красное вино', value=bool(X_test['is_red'].median())))
    wine = pd.DataFrame([values])[features]
    probability = float(model.predict_proba(wine)[0, 1])
    st.metric('Вероятность хорошего вина', f'{probability:.1%}')
    st.write('Решение модели: включить в подборку' if probability >= threshold
             else 'Решение модели: не включать в подборку')

with metrics_tab:
    prediction = (scores >= threshold).astype(int)
    precision = metrics.precision_score(y_test, prediction, zero_division=0)
    recall = metrics.recall_score(y_test, prediction, zero_division=0)
    f1 = metrics.f1_score(y_test, prediction, zero_division=0)
    for column, name, value in zip(st.columns(3), ['Precision', 'Recall', 'F1'], [precision, recall, f1]):
        column.metric(name, f'{value:.3f}')
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
    ax.scatter(fp / (fp + tn), recall, color='crimson', zorder=5, label=f'Порог {threshold:.3f}')
    ax.legend(fontsize=8)
    ax.set_title('ROC кривая')
    middle.pyplot(fig)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(4, 4))
    metrics.PrecisionRecallDisplay.from_predictions(y_test, scores, ax=ax, name='LR')
    ax.scatter(recall, precision, color='crimson', zorder=5, label=f'Порог {threshold:.3f}')
    ax.legend(fontsize=8)
    ax.set_title('PR кривая')
    right.pyplot(fig)
    plt.close(fig)

with batch_tab:
    upload = st.file_uploader('CSV с 12 признаками, без target', type=['csv'])
    if upload is not None:
        try:
            data = pd.read_csv(upload)
            missing = [name for name in features if name not in data.columns]
            if missing:
                st.error('Не хватает столбцов: ' + ', '.join(missing))
            else:
                result = data.copy()
                result['probability_good'] = model.predict_proba(data[features])[:, 1]
                result['prediction'] = (result['probability_good'] >= threshold).astype(int)
                st.dataframe(result, hide_index=True)
                st.download_button(
                    'Скачать прогнозы', result.to_csv(index=False).encode('utf-8-sig'),
                    file_name='wine_predictions.csv', mime='text/csv',
                )
        except (ValueError, pd.errors.ParserError, UnicodeError) as error:
            st.error(f'Не удалось обработать CSV: {error}')

with coefficients_tab:
    coefficients = pd.Series(model[-1].coef_.ravel(), index=features).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(coefficients.index, coefficients, color=np.where(coefficients >= 0, '#568c6b', '#b55d63'))
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set(xlabel='Коэффициент после масштабирования')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    positive = coefficients[coefficients > 0].tail(3).sort_values(ascending=False)
    negative = coefficients[coefficients < 0].head(3)
    st.write('Сильнее всего за хорошее вино: ' + ', '.join(positive.index) + '.')
    st.write('Сильнее всего против: ' + ', '.join(negative.index) + '.')
