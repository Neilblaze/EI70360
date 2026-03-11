import numpy as np

class NaiveBayes():
    def __init__(self, smoothing=1.0):
        self.class_prior = {}
        self.feature_likelihood = {}
        self.smoothing = smoothing

    def forward(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)

        for c in self.classes:
            X_c = X[y == c]
            self.class_prior[c] = X_c.shape[0] / n_samples
            self.feature_likelihood[c] = (X_c.sum(axis=0) + self.smoothing) / (X_c.shape[0] + 2)

    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = []
            
            for c in self.classes:
                prior_log = np.log(self.class_prior[c])
                likelihood_log = np.sum(
                    x * np.log(self.feature_likelihood[c]) + 
                    (1 - x) * np.log(1 - self.feature_likelihood[c]))
                posterior = prior_log + likelihood_log
                posteriors.append(posterior)

            pred = self.classes[np.argmax(posteriors)]
            predictions.append(pred)
        
        return np.array(predictions)
