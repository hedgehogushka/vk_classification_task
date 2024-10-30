import joblib
import pandas as pd

model = joblib.load('best_model.sav')
df_test = pd.read_parquet('df_test_features.parquet')
predictions = model.predict(df_test)
submission = pd.DataFrame({
    'id': df_test['id'],
    'prediction': predictions
})
submission.to_csv('submission.csv', index=False)