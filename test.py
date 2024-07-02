
import joblib

loaded_model = joblib.load(r'C:\Users\ASUS\Desktop\FUN SIMPLY\polynomial\Cell_Phones_and_Accessories_5.json\sentiment_analysis_model.pkl')
result = loaded_model.predict(["it is not great "])

print("The loaded model predicts:", result[0])
