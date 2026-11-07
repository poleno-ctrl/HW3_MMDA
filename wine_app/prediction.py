import csv
from io import StringIO

import numpy as np
import pandas as pd


def read_wines_csv(content: bytes) -> pd.DataFrame:
    text = content.decode('utf-8-sig')
    header = next(csv.reader(StringIO(text)), [])
    if not header:
        raise ValueError('CSV пуст. Загрузите таблицу с заголовками и строками вин.')
    if len(header) != len(set(header)):
        raise ValueError('В CSV повторяются названия столбцов.')
    return pd.read_csv(StringIO(text))


def validate_wines(data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    missing = [name for name in features if name not in data.columns]
    if missing:
        raise ValueError('Не хватает столбцов: ' + ', '.join(missing))
    if data.empty:
        raise ValueError('В CSV нет строк с винами.')
    try:
        wines = data.loc[:, features].apply(pd.to_numeric, errors='raise').astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError('Все признаки должны быть числами; десятичный разделитель — точка.') from error
    if not np.isfinite(wines.to_numpy()).all():
        raise ValueError('Заполните пропуски и замените бесконечные значения в признаках.')
    if not wines['is_red'].isin([0, 1]).all():
        raise ValueError('is_red принимает только 0 (белое) или 1 (красное).')
    return wines


def predict_batch(model, data: pd.DataFrame, features: list[str], threshold: float) -> pd.DataFrame:
    wines = validate_wines(data, features)
    result = data.copy()
    result['probability_good'] = model.predict_proba(wines)[:, 1]
    result['prediction'] = (result['probability_good'] >= threshold).astype(int)
    return result
