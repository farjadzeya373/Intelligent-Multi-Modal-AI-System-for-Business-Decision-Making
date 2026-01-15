from src.data_ingestion import load_data
from src.data_preprocessing import preprocess
from src.model_training import train_model, save_model
from src.model_evaluation import evaluate

df = load_data("data/raw/data.csv")
X_train, X_test, y_train, y_test, scaler = preprocess(df, "target")
model = train_model(X_train, y_train)
save_model(model, scaler)
evaluate(model, X_test, y_test)