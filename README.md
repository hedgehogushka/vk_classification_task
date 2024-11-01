### Структура решения.

Все файлы для решения находятся в папке time_series 

Полный код находится в файле time_series/vk_task.ipynb

Для быстрой генерации решения запустите time_series/general_submission.py

В файле time_series/best_model.sav находится выбранная в процессе обучения модель. 

В файле time_series/df_test_features.parquet находится сгенерированная в процессе обучения таблица признаков для тестовых данных. 

Решение находится в файле time_series/submission.csv

### Описание решения.
Я загружаю тестовые и тренировочные данные, находящиеся в parquet-файлах. После небольшого визуального анализа полученных временных рядов, я вычисляю характеризующие их признаки и размечаю их.
Разбиваю данные для обучения на тренировочную и валидационную выборку, пишу функцию для более удобного вычисления roc-auc метрики, и начинаю подбирать наилучшую модель.

В ходе решения рассматриваю на различных гиперпараметрах следующие модели:
Random Forest
XGBoost 
CatBoost 
Logistic Regression 

Останавливаюсь на стекинге на перечисленных выше моделях. 
Аналогично размечаю тестовые данные, запускаю их на выбранной модели и сохраняю полученный результат в submission.csv. Также сохраняю наилучшую модель в best_model.sav и набор признаков тестовых данных df_test_features.parquet, чтобы не генерировать их при повторном запуске. 
