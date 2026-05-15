## README: Categorical Data Analysis (STAT 4104)

This repository contains the comprehensive laboratory assignment for the course **STAT 4104: Categorical Data Analysis**, supervised by **Assoc. Prof. Dr. Md. Siddikur Rahman**. The project implements a wide range of statistical modeling techniques using **R** and **Python** to analyze real-world datasetas across public health, education, and social science domains.

---

## 📂 Project Overview

The assignment is divided into 11 core problems, moving from basic exploratory data analysis (EDA) to advanced generalized linear models (GLMs) for count and categorical data.

### **Section 1: Bivariate & Multivariate Analysis**

* **Problem 1: Lung Disease Determinants (Python)**
* **Techniques:** EDA (Skewness, Prevalence), Chi-square tests, Odds Ratios, and Logistic Regression.


* **Focus:** Identifying key lifestyle predictors (Smoking, Pollution, Income) and interpreting interaction effects.




* **Problem 2: One-Way ANOVA**
* **Scenario:** Testing weight loss efficiency across three exercise programs (A, B, and C) using 90 subjects.


* **Tasks:** Assumption checking and post-hoc treatment difference analysis.




* **Problem 3: Nigeria Demographic & Health Survey (NDHS)**
* **Scenario:** Factors affecting anemia levels in children aged 0-59 months.


* **Techniques:** Contingency tables, Chi-square tests, and contribution diagrams.




* **Problem 4: Fisher's Exact Test**
* **Scenario:** Testing drug treatments (A, B, C) vs. disease outcomes when cell counts are small.


* **Tasks:** Fisher’s Exact Test and post-hoc pairwise comparisons.





### **Section 2: Generalized Linear Models (GLMs)**

* **Problem 5: Binary Logistic Regression**
* **Dataset:** UCLA Graduate Admissions (GRE, GPA, Prestige).


* **Focus:** Modeling binary outcomes (Admit/Don't Admit).




* **Problem 6: Poisson Regression**
* **Dataset:** High school student awards.


* **Focus:** Modeling count data where the mean equals the variance.




* **Problem 7: Negative Binomial Regression**
* **Dataset:** Student attendance behavior (Days absent).


* **Focus:** Handling **overdispersion** in count data.





### **Section 3: Specialized Count Models**

* **Problem 8 & 9: Zero-Inflated Models (ZIP & ZINB)**
* **Dataset:** Wildlife fish catch data.


* **Focus:** Modeling "excess zeros" caused by visitors who did not fish vs. those who fished but caught nothing.




* **Problem 10: Zero-Truncated Poisson (ZTP)**
* **Dataset:** Hospital length of stay (minimum stay of 1 day).


* **Focus:** Modeling count data where the value zero is impossible.




* **Problem 11: Zero-Truncated Negative Binomial (ZTNB)**
* **Focus:** Combining zero-truncation with overdispersion adjustments for hospital stay data.





---

## 🛠️ Requirements & Tools

* **Python:** `pandas`, `numpy`, `statsmodels`, `scipy`, `sklearn`, `matplotlib`.


* **R:** `MASS`, `pscl`, `ggplot2`, `stats`.


* **Datasets:** Sourced from UCLA IDRE, Kaggle, and NDHS 2018.



---

## 📈 Key Findings & Learning Objectives

1. **Model Selection:** Understanding when to use Poisson vs. Negative Binomial based on dispersion.


2. **Structural Zeros:** Implementing Zero-Inflated models for complex behavioral data.


3. **Risk Interpretation:** Using Odds Ratios (OR) and Incidence Rate Ratios (IRR) to communicate health and social findings.


4. **Diagnostics:** Evaluating model performance via ROC-AUC, confusion matrices, and residual analysis.
