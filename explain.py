import shap
import matplotlib.pyplot as plt

def explain_model(model, vectorizer, X_train, X_test):
    # Convert to dense (small subset only)
    X_train_dense = X_train[:50].toarray()   # 🔥 reduce size
    X_test_dense = X_test[:10].toarray()     # 🔥 reduce size

    feature_names = vectorizer.get_feature_names_out()

    # Use small background
    background = X_train_dense[:20]

    # KernelExplainer
    explainer = shap.KernelExplainer(model.predict_proba, background)

    # Limit samples (IMPORTANT)
    shap_values = explainer.shap_values(X_test_dense, nsamples=50)

    shap.summary_plot(shap_values, X_test_dense, feature_names=feature_names)

    plt.savefig("outputs/shap_plot.png")
    plt.close()