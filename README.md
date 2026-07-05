FinTech Innovations — Loan Approval Risk Model

A machine learning pipeline that predicts loan approval decisions for FinTech Innovations' Risk Analytics team, built to replace/support a manual, inconsistent loan review process with a cost-aware, data-driven recommendation engine.
A false positive is 6.25x more expensive than a false negative — so the model isn't just tuned for accuracy, it's tuned to minimize actual dollar cost. After tuning a Gradient Boosting classifier and calibrating its decision threshold against these real costs, the model cuts total error cost on the test set from $5.45M (default threshold) to $3.81M (cost-optimal threshold) — a 30% improvement from threshold tuning alone, and a large improvement over the naive "deny everyone" baseline (~$7.65M on the same test set).

Business problem
FinTech Innovations partners with banks to modernize loan approval. The current manual process is slow, inconsistent between reviewers, and has no formal way to weigh "reject a good applicant" against "approve a bad one." This project builds a classifier that predicts LoanApproved from applicant data, with the explicit goal of minimizing total dollar cost rather than maximizing accuracy.

Why classification over regression: the dataset also contains a continuous RiskScore, which could support a regression approach. We chose classification because the $8,000/$50,000 costs are naturally a confusion-matrix problem (they're defined per decision, not per risk unit), and a classifier maps directly onto the approve/deny action loan officers currently take. predict_proba still exposes an underlying risk-like probability for human judgment on borderline cases.

Dataset
20,000 historical loan applications, each with ~34 fields covering applicant demographics (age, employment, education, marital status), financial position (income, assets, liabilities, net worth, savings), credit history (credit score, previous defaults, credit inquiries, utilization), and loan terms (amount, duration, interest rate, purpose). Target: LoanApproved (binary), with RiskScore available as a secondary signal (dropped from model inputs — it's a near-perfect proxy for the label and would leak information).

Class balance: ~24% approved / 76% denied.

Methodology

The notebook follows CRISP-DM:


Business Understanding — stakeholder analysis, cost-of-error framing, classification vs. regression justification, custom cost metric definition, baseline targets.
Data Understanding — 10 visualizations (distributions, correlation heatmap, pairplot, categorical breakdowns, missing-value pattern), two statistical tests (Welch t-tests, chi-square), feature typing and missing-value analysis.
Data Preparation — a ColumnTransformer + Pipeline with separate branches for numeric (median impute + scale), categorical (most-frequent impute + one-hot), and ordinal (most-frequent impute + ordinal encode) features, a custom FunctionTransformer for engineered ratios (disposable income, asset-to-liability ratio, savings-to-income ratio), and SelectFromModel feature selection — all inside one deployable pipeline.
Modeling — 5 algorithms compared by cross-validation (Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting), scored on a custom business-cost metric and ROC-AUC. The top candidates go through a two-stage hyperparameter search (RandomizedSearchCV → GridSearchCV, tuning both model and preprocessing parameters), plus a voting ensemble for comparison.
Evaluation — confusion matrix and ROC-AUC at the default threshold, then a cost-vs-threshold sweep to find the dollar-optimal cutoff, a baseline-vs-model dollar impact table, performance broken down by applicant segment, feature importance with business interpretation, and a bias/limitations discussion.


Key results


Selected model: Gradient Boosting (tuned), outperforming Logistic Regression, KNN, Decision Tree, Random Forest, and an RF+GB voting ensemble on the business-cost metric.
Test ROC-AUC: ~0.986
Default threshold (0.5) cost: $5,454,000 on the test set
Cost-optimal threshold (~0.83) cost: $3,812,000 — a 30% reduction from threshold tuning alone
Top predictive features: disposable income, total debt-to-income ratio, and interest rate offered — all direct measures of repayment capacity rather than demographic proxies.


Key takeaway

The biggest lever in this project wasn't algorithm choice — it was calibrating the decision threshold to the actual cost structure. A model can have excellent AUC and still lose the business money at the default 0.5 cutoff, because that cutoff implicitly treats a $50,000 mistake and an $8,000 mistake as equally bad. Threshold choice should be owned as a business policy decision (revisited if the underlying cost figures change), not left as a technical default.

Limitations


LoanApproved reflects historical manual decisions, not confirmed loan defaults — the model can inherit any bias already present in that process. Recommended as a decision-support tool for loan officers, not full automation.
Cost figures ($8,000/$50,000) are business estimates applied to a proxy label; validate against real default outcomes once available.
Performance varies somewhat across applicant segments (employment status, education) — a fair-lending review is recommended before deployment.
