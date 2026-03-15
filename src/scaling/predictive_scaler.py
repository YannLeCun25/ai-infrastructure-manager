import numpy as np
from sklearn.linear_model import LinearRegression

class PredictiveScaler:
    """Predicts future GPU demand using time-series linear trends."""
    def __init__(self):
        self.model = LinearRegression()
        self.history = []

    def update_metrics(self, timestamp: float, usage: float):
        self.history.append([timestamp, usage])

    def forecast(self, next_timestamp: float) -> float:
        if len(self.history) < 2: return 0.0
        X, y = np.array(self.history)[:, 0].reshape(-1, 1), np.array(self.history)[:, 1]
        self.model.fit(X, y)
        return self.model.predict([[next_timestamp]])[0]
