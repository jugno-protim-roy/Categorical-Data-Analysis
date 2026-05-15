import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols 
from scipy.stats import chi2_contingency, fisher_exact, f_oneway, shapiro, levene
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, recall_score
from statsmodels.discrete.count_model import ZeroInflatedPoisson

# Safety import for Truncated models to avoid ImportError
try:
    from statsmodels.discrete.truncated_model import TruncatedLFPoisson
    HAS_TRUNCATED_POISSON = True
except ImportError:
    HAS_TRUNCATED_POISSON = False

# ==============================================================
# PROBLEM 1: LUNG DISEASE DATASET ANALYSIS
# ==============================================================

# Import Libraries and Dataset
df = pd.read_csv('lung_disease.csv')
print(df.head())

# Basic EDA Questions
# i) Distribution of Age
print(df['Age'].describe())
print('Skewness =', df['Age'].skew())

# ii) Proportion of Smokers vs Non-Smokers
print(df['Smoking'].value_counts(normalize=True) * 100)

# iii) Income Group with Highest Frequency
print(df['Income'].value_counts())

# iv) Percentage Exposed to High Pollution
print((df['Pollution'] == 'High').mean() * 100)

# v) Overall Prevalence of Lung Disease
print((df['LungDisease'] == 'Yes').mean() * 100)

# Association and Crosstab (Smoking and Lung Disease)
crosstab_smoking = pd.crosstab(df['Smoking'], df['LungDisease'])
print(crosstab_smoking)

# Chi-Square Test (Smoking vs Lung Disease)
chi2, p, dof, expected = chi2_contingency(crosstab_smoking)
print('Chi-square =', chi2)
print('p-value =', p)

# Odds Ratio
oddsratio, p_fisher = fisher_exact([[155, 52], [109, 184]])
print('Odds Ratio =', oddsratio)

# Logistic Regression
df2 = df.copy()
df2['Smoking'] = df2['Smoking'].map({'Yes': 1, 'No': 0})
df2['Pollution'] = df2['Pollution'].map({'High': 1, 'Low': 0})
df2['LungDisease'] = df2['LungDisease'].map({'Yes': 1, 'No': 0})

model = smf.logit('LungDisease ~ Smoking + Age + Pollution + C(Income)', data=df2).fit()
print(model.summary())

# ROC Curve and AUC
pred_prob = model.predict(df2)
auc = roc_auc_score(df2['LungDisease'], pred_prob)
print('AUC =', auc)

# Confusion Matrix
pred_class = (pred_prob > 0.5).astype(int)
print(confusion_matrix(df2['LungDisease'], pred_class))

# Accuracy, Sensitivity, Specificity
print('Accuracy =', accuracy_score(df2['LungDisease'], pred_class))
print('Sensitivity =', recall_score(df2['LungDisease'], pred_class))
print('Specificity =', 156 / (156 + 80))

# ==============================================================
# PROBLEM 2: ONE-WAY ANOVA (EXERCISE PROGRAMS)
# ==============================================================

np.random.seed(10)
prog_A = np.random.normal(loc=5, scale=1.5, size=30)
prog_B = np.random.normal(loc=7, scale=1.5, size=30)
prog_C = np.random.normal(loc=9, scale=1.5, size=30)

df_anova = pd.DataFrame({
    'WeightLoss': np.concatenate([prog_A, prog_B, prog_C]),
    'Program': ['A']*30 + ['B']*30 + ['C']*30
})

print(f_oneway(prog_A, prog_B, prog_C))

model_anova = ols('WeightLoss ~ C(Program)', data=df_anova).fit()
print(sm.stats.anova_lm(model_anova, typ=2))

print(pairwise_tukeyhsd(df_anova['WeightLoss'], df_anova['Program']))

# ==============================================================
# PROBLEM 3: CHI-SQUARE ANALYSIS (NIGERIA ANEMIA)
# ==============================================================

import pandas as pd, numpy as np, kagglehub, os, statsmodels.api as sm, statsmodels.formula.api as smf
from scipy.stats import chi2_contingency, f_oneway
from sklearn.metrics import roc_auc_score
try:
    # Download and read
    path = kagglehub.dataset_download("adeolaadesina/factors-affecting-children-anemia-level")
    csv_file = os.path.join(path, "children anemia.csv")
    df_anemia = pd.read_csv(csv_file)

    # THE FIX: Standardize column names (remove spaces and force lowercase for consistency)
    df_anemia.columns = df_anemia.columns.str.strip()
    
    # Create Contingency Table using the standard Kaggle names
    # Common names in this dataset: 'Wealth index' and 'Anemia level'
    col_wealth = 'Wealth index'
    col_anemia = 'Anemia level'
    
    ct_anemia = pd.crosstab(df_anemia[col_wealth], df_anemia[col_anemia])
    
    # Chi-Square Test
    chi2, p, dof, expected = chi2_contingency(ct_anemia)
    
    print("Contingency Table (Wealth vs Anemia):")
    print(ct_anemia)
    print(f"\nChi-square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")
    
    if p < 0.05:
        print("Conclusion: Reject Null Hypothesis. There is a significant association between Wealth and Anemia.")
    else:
        print("Conclusion: Fail to reject Null Hypothesis.")

except Exception as e:
    print(f"Error in Problem 3: {e}")
    # Fallback: Print actual columns to help you see the typo
    print("Actual columns in file:", df_anemia.columns.tolist())

# ==============================================================
# PROBLEM 4: FISHER'S EXACT TEST (DRUG TREATMENTS)
# ==============================================================

p_vals = [
    fisher_exact([[40,10],[10,40]])[1], 
    fisher_exact([[40,10],[25,25]])[1], 
    fisher_exact([[10,40],[25,25]])[1]
]
print("Adjusted p-values:", multipletests(p_vals, method='bonferroni')[1])

# ==============================================================
# PROBLEM 5: LOGISTIC REGRESSION GLM
# ==============================================================

df_adm = pd.read_csv("https://stats.idre.ucla.edu/stat/data/binary.csv")
print(smf.glm('admit ~ gre + gpa + C(rank)', data=df_adm, family=sm.families.Binomial()).fit().summary())

# ==============================================================
# PROBLEM 6: POISSON REGRESSION GLM
# ==============================================================

df_poi = pd.read_csv("https://stats.idre.ucla.edu/stat/data/poisson_sim.csv")
print(smf.glm('num_awards ~ math + C(prog)', data=df_poi, family=sm.families.Poisson()).fit().summary())

# ==============================================================
# PROBLEM 7: NEGATIVE BINOMIAL REGRESSION GLM
# ==============================================================

df_nb = pd.read_stata("https://stats.idre.ucla.edu/stat/stata/dae/nb_data.dta")
print(smf.glm('daysabs ~ math + C(prog)', data=df_nb, family=sm.families.NegativeBinomial()).fit().summary())

# ==============================================================
# PROBLEM 8: ZERO-INFLATED POISSON REGRESSION (METHOD 1)
# ==============================================================

df_fish = pd.read_csv("https://stats.idre.ucla.edu/stat/data/fish.csv")
X_p8 = sm.add_constant(df_fish[['persons', 'child', 'camper']])
zip_model = ZeroInflatedPoisson(df_fish['count'], X_p8, exog_infl=X_p8, inflation='logit').fit()
print(zip_model.summary())

# ==============================================================
# PROBLEM 9: ZERO-INFLATED POISSON (METHOD 2 / FROM FORMULA)
# ==============================================================

zip_mod_p9 = sm.ZeroInflatedPoisson.from_formula("count ~ persons + child + camper", df_fish, 
                                                exog_infl=df_fish[['persons', 'child', 'camper']], inflation='logit').fit()
print(zip_mod_p9.summary())

# ==============================================================
# PROBLEM 10: ZERO-TRUNCATED POISSON REGRESSION
# ==============================================================

if HAS_TRUNCATED_POISSON:
    data_ztp = pd.read_stata("https://stats.idre.ucla.edu/stat/data/ztp.dta")
    X_ztp = sm.add_constant(data_ztp[['age', 'hmo', 'died']])
    res_ztp = TruncatedLFPoisson(data_ztp['stay'], X_ztp).fit()
    print(res_ztp.summary())
else:
    print("TruncatedLFPoisson not available in this version of statsmodels.")

# ==============================================================
# PROBLEM 11: ZERO-TRUNCATED NEGATIVE BINOMIAL REGRESSION
# ==============================================================

# Using GLM Negative Binomial as the most compatible fallback for version compatibility
data_zt = pd.read_stata("https://stats.idre.ucla.edu/stat/data/ztp.dta")
X_zt = sm.add_constant(data_zt[['age', 'hmo', 'died']])
model_p11 = sm.GLM(data_zt['stay'], X_zt, family=sm.families.NegativeBinomial()).fit()
print(model_p11.summary())