from lime.lime_text import LimeTextExplainer

def explain_prediction(model, vectorizer, text):
    explainer = LimeTextExplainer(class_names=["FAKE", "REAL"])
    pred_fn = lambda x: model.predict_proba(vectorizer.transform(x))
    exp = explainer.explain_instance(text, pred_fn, num_features=10)
    exp.show_in_notebook()
    # For Streamlit: st.components.v1.html(exp.as_html(), height=600)
