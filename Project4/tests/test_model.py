from predictor.model import get_models

def test_model_setup():
    models = get_models()
    assert "Linear Regression" in models
    assert "Decision Tree" in models

f __name__ == '__main__':
    test_model_setup()
    print("test_model_setup passed."
