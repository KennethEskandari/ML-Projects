from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

def get_models():
    models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(max_depth=5)
            }

    return models


