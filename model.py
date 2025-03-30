import numpy as np
from hmmlearn import hmm
from sklearn.mixture import GaussianMixture

def train_hmm(data: list):
    X = np.array([[d["Temperature"], d["Pressure"], d["Vibration"]] for d in data])
    model = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=100)
    model.fit(X[:3])  # 정상 구간만 학습
    return model, X

def detect_anomaly(model, X):
    log_likelihoods = model._compute_log_likelihood(X)  # shape: (n_samples, n_components)
    frame_scores = np.max(log_likelihoods, axis=1)      # 각 시점의 최대 log-likelihood

    baseline = np.mean(frame_scores[:3])  # 초기 3개를 정상 기준
    threshold = baseline - 5

    for i, score in enumerate(frame_scores):
        if score < threshold:
            return i, score

    return None, None
