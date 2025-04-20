from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from predictor.data_loader import load_data
from predictor.features import prepare_features
from predictor.model import get_models

def train_and_evaluate():
    df = load_data()
    x,y = prepare_features(df)

    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)

    models = get_models()
    for name, mode in models.items():
        mode.fit(x_train, y_train)
        predictions = model.predict(x_test)
        rmse = mean_squared_error(y_test, predictions, squared = False)
        print(f"(name) RMSE : (rmse: 2f)")

    if __name__ == '__main__':
        train_and_evaluate()
