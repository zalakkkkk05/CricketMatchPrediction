import pickle
import pandas as pd

def test_model_prediction():
    # Load model & features
    model = pickle.load(open("src/models/pickles/stacked_model.pkl", "rb"))
    features = pickle.load(open("src/models/pickles/feature_columns.pkl", "rb"))

    # Create dummy input with expected features
    dummy = pd.DataFrame(0, index=[0], columns=features)
    dummy.iloc[0, :8] = 0.5  # dummy numeric features

    # Try prediction
    prediction = model.predict(dummy)[0]

    # Assert prediction is an integer (encoded label)
    assert isinstance(prediction, (int, np.integer)), f"Unexpected prediction type: {type(prediction)}"
