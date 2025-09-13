import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, auc


st.set_page_config(page_title="Student Well-being Dashboard", layout="wide")

@st.cache_data
def load_data(s):
    return pd.read_csv(s)

def run_cv(model, X, y, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
    mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
    mse_scores = cross_val_score(model, X, y, cv=kf, scoring=mse_scorer)

    return r2_scores, -mse_scores

df = load_data("output.csv")
dfn = load_data("data_null_removed.csv")
dfn['High_Stress'] = (dfn['stress_lv'] >= 4).astype(int)
dfn['Sleep_Deprived'] = (dfn['sleep_hour'] <= 5).astype(int)
dfn['High_Anxiety'] = (dfn['anxiety_lv'] >= 4).astype(int)

label = {
    'year' : "Year",
    'gender' : "Gender",
    'age' : "Age",
    'cgpa' : "CGPA",
    'study_hour' : "Study Hours per Week",
    'credit' : "Credits enrolled(in current semester)",
    'extra_curricular' : "Has participation in Extra Curricular activities",
    'extra_curricular_hour' : "Extra Curricular work Hours per Week",
    'job' : 'Has a Job',
    'job_hour' : 'Job work Hours per Week',
    'stress_lv' : "Stress Level",
    'anxiety_lv' : "Anxiety Level",
    'sleep_hour' : "Average Sleep Hours per Night",
    'sleep_lv' : "Quality of Sleep"
}
label_to_col = {v: k for k, v in label.items()}

st.title("Student Workload & Mental Health Dashboard")

with st.sidebar:
    st.header("🔍 Filters")
    years = st.multiselect("Year of Study", sorted(df["year"].unique()), default=sorted(df["year"].unique()))
    genders = st.multiselect("Gender", sorted(df["gender"].unique()), default=sorted(df["gender"].unique()))
    jobs = st.multiselect("Job Status", sorted(df["job"].unique()), default=sorted(df["job"].unique()))
    ext_curricular = st.multiselect("Extra Curricular Activities", sorted(df["extra_curricular"].unique()), default=sorted(df["extra_curricular"].unique()))
    cgpa_min = st.slider("CGPA Min", min_value=0.0, max_value=4.0, value=0.0, step=0.01)
    cgpa_max = st.slider("CGPA Max", min_value=0.0, max_value=4.0, value=4.0, step=0.01)
    
df_f = df[
    (df["year"].isin(years)) & 
    (df["gender"].isin(genders)) & 
    (df["job"].isin(jobs)) & 
    (df["extra_curricular"].isin(ext_curricular)) &
    (df["cgpa"] >= cgpa_min) &
    (df["cgpa"] <= cgpa_max) &
    (cgpa_min <= cgpa_max)
]


st.subheader("Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Number of Students", len(df_f))
c2.metric("Avg Study Hours", f"{df_f['study_hour'].mean():.1f}")
c3.metric("Avg Sleep Hours", f"{df_f['sleep_hour'].mean():.1f}")
c4.metric("Avg Stress (1-5)", f"{df_f['stress_lv'].mean():.2f}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Visualizations", "Distributions", "Regression Models", "Insights", "Correlation Explorer"])

with tab1:
    st.markdown("### Scatterplot")
    col1, col2 = st.columns(2)
    with col1:
        x_var = label_to_col[st.selectbox("Select X-axis", [label[col] for col in ["study_hour", "sleep_hour", "credit", "job_hour"]])]
    with col2:    
        y_var = label_to_col[st.selectbox("Select Y-axis", [label[col] for col in ["stress_lv", "anxiety_lv", "sleep_lv"]])]
    fig1 = px.scatter(df_f, x=x_var, y=y_var, trendline="ols", title=f"{label[y_var]} vs {label[x_var]}", labels = label)
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    st.markdown("### Stress & Anxiety Distributions")
    fig3 = px.histogram(df_f, x="stress_lv", nbins=5, color="year", barmode="overlay", title="Stress Distribution")
    st.plotly_chart(fig3, use_container_width=True)

    fig4 = px.violin(df_f, x="year", y="anxiety_lv",points = 'all', color="gender", box=True, title="Anxiety by Year & Gender")
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("### High Stress Proportion (≥4/5)")
    df_f["high_stress"] = (df_f["stress_lv"] >= 4).astype(int)
    fig5 = px.pie(df_f, names="job", values="high_stress", title="High Stress % by Job Status")
    st.plotly_chart(fig5, use_container_width=True)

with tab3:
    model_type = st.radio("Choose Regression Type", ["Simple Regression", "Multiple Regression", "Polynomial Regression", "Logistic Regression"])
    if(model_type == "Simple Regression"):
        st.title("Simple Regression Explorer")
        pair = [
            ('study_hour', 'stress_lv'),
            ('sleep_hour', 'anxiety_lv'),
            ('job_hour', 'sleep_lv'),
        ]   
        for x, y_col in pair:
            X = dfn[[x]]  
            y = dfn[y_col]  

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2_cv_scores, mse_cv_scores = run_cv(model, X, y, k=5)

            st.subheader(f"Linear Regression: {x} vs {y.name}")
            st.write("**Equation:**")
            st.latex(f"{y.name} = {model.coef_[0]:.4f} \\cdot {x} + {model.intercept_:.4f}")

            st.write("**Metrics on Test Set:**")
            st.write(f"- Coefficient: {model.coef_[0]:.4f}")
            st.write(f"- Intercept: {model.intercept_:.4f}")
            st.write(f"- Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.4f}")
            st.write(f"- R² Score: {r2_score(y_test, y_pred):.4f}")

            st.write("**Cross-Validation (5-fold) Results:**")
            st.write(f"- R² Scores: {r2_cv_scores}")
            st.write(f"- Mean R²: {r2_cv_scores.mean():.4f} ± {r2_cv_scores.std():.4f}")
            st.write(f"- MSE Scores: {mse_cv_scores}")
            st.write(f"- Mean MSE: {mse_cv_scores.mean():.4f} ± {mse_cv_scores.std():.4f}")

            fig, ax = plt.subplots(figsize=(6, 4))  
            fig.tight_layout()
            sns.scatterplot(x=X_test[x], y=y_test, color='blue', label='Test Data', ax=ax)
            sns.lineplot(x=X_test[x], y=y_pred, color='red', label='Regression Line', ax=ax)
            ax.set_title(f"{x} vs {y.name} (R²: {r2_score(y_test, y_pred):.3f}, MSE: {mean_squared_error(y_test, y_pred):.3f})")
            ax.set_xlabel(x)
            ax.set_ylabel(y.name)
            ax.grid(True, alpha=0.3)
            ax.legend()

            st.pyplot(fig)
            st.markdown("---")
            
    elif(model_type == "Multiple Regression"):
        st.title("Multiple Linear Regression Explorer")

        predictors = ['study_hour', 'sleep_hour', 'extra_curricular_hour', 'job_hour', 'credit']
        target = 'stress_lv'

        X = dfn[predictors]
        y = dfn[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.subheader("Multiple Regression Results (Train/Test Split)")
        st.write("**Intercept:**", model.intercept_)
        st.write("**Coefficients:**")
        coef_df = {pred: coef for pred, coef in zip(predictors, model.coef_)}
        st.write(coef_df)

        st.write("**Evaluation Metrics:**")
        st.write(f"- MSE: {mean_squared_error(y_test, y_pred):.4f}")
        st.write(f"- R² Score: {r2_score(y_test, y_pred):.4f}")

        r2_cv_scores, mse_cv_scores = run_cv(model, X, y, k=5)
        st.subheader("Cross-Validation Results (5-fold)")
        st.write(f"- R² Scores: {r2_cv_scores}")
        st.write(f"- Mean R²: {r2_cv_scores.mean():.4f} ± {r2_cv_scores.std():.4f}")
        st.write(f"- MSE Scores: {mse_cv_scores}")
        st.write(f"- Mean MSE: {mse_cv_scores.mean():.4f} ± {mse_cv_scores.std():.4f}")

        st.subheader("Correlation Matrix of Predictors")
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.tight_layout()
        sns.heatmap(X.corr(), annot=True, cmap="plasma", fmt=".2f", ax=ax)
        ax.set_title("Correlation Matrix of Predictors")
        st.pyplot(fig)
        features = ["study_hour", "credit", "sleep_lv"]
        
    elif(model_type == "Polynomial Regression"):
        st.title("Polynomial Regression Explorer")
        cases = [
            ("sleep_hour", "anxiety_lv"),
            ("study_hour", "stress_lv"),
            ("job_hour", "sleep_lv"),
        ]

        degree = 4

        for predictor, target in cases:
            X = dfn[[predictor]].values
            Y = dfn[target].values

            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

            poly = PolynomialFeatures(degree=degree)
            X_train_poly = poly.fit_transform(X_train)
            X_test_poly = poly.transform(X_test)

            model = LinearRegression()
            model.fit(X_train_poly, y_train)
            y_pred = model.predict(X_test_poly)

            st.subheader(f"Polynomial Regression ({predictor} → {target})")
            st.write(f"**Degree:** {degree}")
            st.write(f"- R² (Test Set): {r2_score(y_test, y_pred):.4f}")
            st.write(f"- MSE (Test Set): {mean_squared_error(y_test, y_pred):.4f}")

            X_poly_full = poly.fit_transform(X)  # full dataset for CV
            r2_cv_scores, mse_cv_scores = run_cv(model, X_poly_full, Y, k=5)
            st.write("**Cross-Validation (5-fold):**")
            st.write(f"- R² Scores: {r2_cv_scores}")
            st.write(f"- Mean R²: {r2_cv_scores.mean():.4f} ± {r2_cv_scores.std():.4f}")
            st.write(f"- MSE Scores: {mse_cv_scores}")
            st.write(f"- Mean MSE: {mse_cv_scores.mean():.4f} ± {mse_cv_scores.std():.4f}")

            fig, ax = plt.subplots(figsize=(6, 4))
            fig.tight_layout()
            ax.scatter(X_test, y_test, color='blue', label='Test Data')

            sorted_idx = X_test.ravel().argsort()
            ax.plot(X_test.ravel()[sorted_idx], y_pred[sorted_idx], color='red', label='Polynomial Fit')
            
            ax.set_xlabel(predictor)
            ax.set_ylabel(target)
            ax.set_title(f"Polynomial Regression: {predictor} vs {target}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            st.markdown("---")
        
    else:
        st.title("Logistic Regression Explorer")

        pair = [
            ('study_hour', 'High_Stress'),
            ('sleep_hour', 'High_Anxiety'),
            ('job_hour', 'Sleep_Deprived')
        ]

        for x, y_col in pair:
            X = dfn[[x]]
            y = dfn[y_col]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            st.subheader(f"Logistic Regression: {x} vs {y_col}")
            st.write("**Coefficient:**", model.coef_[0][0])
            st.write("**Intercept:**", model.intercept_[0])
            st.write("**Accuracy (Test Set):**", accuracy_score(y_test, y_pred))

            st.write("**Classification Report:**")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.json(report)

            st.write("**Confusion Matrix:**")
            cm = confusion_matrix(y_test, y_pred)
            st.write(cm)

            fpr, tpr, thresholds = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            fig, ax = plt.subplots(figsize=(6, 4))
            fig.tight_layout()
            ax.plot(fpr, tpr, color='blue', label=f"ROC curve (AUC = {roc_auc:.3f})")
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC Curve: {x} vs {y_col}")
            ax.legend()
            ax.grid(alpha=0.3)

            st.pyplot(fig)
            st.markdown("---")
        
with tab4:
    st.header("Key Insights From Filtered Data")
    avg_stress = df_f["stress_lv"].mean()
    avg_sleep = df_f["sleep_hour"].mean()
    avg_study = df_f["study_hour"].mean()
    st.write(f"- Among **{len(df_f)} students**, the average stress is **{avg_stress:.2f}**.")
    st.write(f"- They study about **{avg_study:.1f} hours/week** and sleep about **{avg_sleep:.1f} hours/night**.")
    if avg_study > 30:
        st.warning("⚠️ High workload: Students with more than 30 study hours tend to report higher stress.")
    if avg_sleep < 6:
        st.warning("😴 Low sleep: Students sleeping less than 6 hours show higher anxiety levels.")
    if avg_stress >= 4:
        st.error("🚨 Warning: Many students are at high stress levels (≥4/5).")
    else:
        st.success("✅ Stress levels are moderate on average.")
    
    st.header("Key Insights From Survay Data")

    insights = [
        "Many features showed **outliers** and **skewed distributions**, but preprocessing (null handling, median imputation) improved correlations.",
        "**Stress, anxiety, and sleep quality** are closely related: higher stress usually comes with higher anxiety and poorer sleep.",
        "**Year of study** strongly correlates with **age**, but anxiety does *not* differ significantly across years.",
        "Lifestyle/academic factors (study hours, part-time jobs, activities, credits) do **not significantly affect stress** in most cases.",
        "**Credits taken** showed a *small but significant* positive correlation with anxiety — students taking more credits tend to feel slightly more anxious.",
        "Regression models (linear, polynomial, multiple) explain **very little variance**: study and sleep hours have only weak effects on stress/anxiety.",
        "Logistic regression models mostly fail due to **class imbalance** (predicting everyone as “high stress” or “anxious”), even if accuracy looks high.",
        "**Part-time work hours** are *not strong predictors* of stress, sleep, or anxiety."
    ]

    for point in insights:
        st.markdown(f"- {point}")
    
with tab5:
    st.markdown("### Correlation Explorer")
    all_columns = list(label.keys())
    selected_labels = st.multiselect("Choose variables", [label[col] for col in all_columns],
                                  default=[label["study_hour"],label["sleep_hour"],label["stress_lv"]])
    selected_vars = [label_to_col[sel] for sel in selected_labels]
    if len(selected_vars) >= 2:
        corr = dfn[selected_vars].corr()
        corr.rename(index=label,columns=label,inplace=True)
        fig2 = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap",color_continuous_scale="plasma")
        fig2.update_layout(height = 700)
        st.plotly_chart(fig2, use_container_width=True)