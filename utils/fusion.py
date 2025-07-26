def fuse_predictions(predictions):
    avg_score = sum(predictions) / len(predictions)
    verdict = "Likely Deepfake" if avg_score > 0.5 else "Likely Real"
    return round(avg_score, 2), verdict
