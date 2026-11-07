Версия Python 3.14.7.

Команды запуска из папки wine_app:

python3.14 -m pip install -r requirements.txt

python3.14 -m streamlit run app.py

Для модели используется st.cache_resource, потому что он сохраняет один общий объект. Для таблиц используется st.cache_data, потому что он возвращает отдельные копии данных. Без декоратора joblib.load будет заново загружать модель при каждом действии пользователя.
