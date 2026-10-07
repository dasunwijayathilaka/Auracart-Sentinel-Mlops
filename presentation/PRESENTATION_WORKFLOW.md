# AuraCart Sentinel — Presentation Workflow & Slide Plan

**Module:** ITS 2140 — Machine Learning  
**Group:** CoreGenix | GDSE 69 (Panadura)  
**Project:** AuraCart Sentinel — Production-Grade E-Commerce Analytics & MLOps System  
**Date:** April 2026

---

## 📋 TABLE OF CONTENTS

1. [Presentation Rules & Constraints](#1-presentation-rules--constraints)
2. [Team Assignment & Time Allocation](#2-team-assignment--time-allocation)
3. [Slide-by-Slide Breakdown](#3-slide-by-slide-breakdown)
   - [Presenter 1 — Dilusha Sandaruwan](#presenter-1--dilusha-sandaruwan-karunathilaka-leader)
   - [Presenter 2 — Madusha Lakmina](#presenter-2--madusha-lakmina)
   - [Presenter 3 — Harsha Dilan](#presenter-3--harsha-dilan)
   - [Presenter 4 — Thanura Vipulanga](#presenter-4--thanura-vipulanga)
   - [Presenter 5 — Chamod Thejan](#presenter-5--chamod-thejan)
   - [Presenter 6 — Dasun Wijayathilaka](#presenter-6--dasun-wijayathilaka)
4. [Viva Voce Preparation Guide](#4-viva-voce-preparation-guide)
5. [Slide Design Guidelines](#5-slide-design-guidelines)
6. [Rehearsal Checklist](#6-rehearsal-checklist)
7. [Available Figures & Screenshots](#7-available-figures--screenshots)

---

## 1. PRESENTATION RULES & CONSTRAINTS

> **Source:** *ITS 2140 Group Project Presentation Guidelines (GDSE 69/70)*

### Time Budget

| Component              | Duration    |
|------------------------|-------------|
| **Formal Presentation** | **30 minutes** (strictly enforced) |
| **Viva Voce (Q&A)**    | **30 minutes** (~5 min per member) |
| **Total Session**       | **60 minutes** |

### Mandatory Rules

1. **ALL 6 members MUST present** — no exceptions.
2. **No code on slides** — all code belongs in Jupyter notebooks. The presentation is for a *non-technical business audience*.
3. **Focus on visuals** — charts, graphs, and data visualizations must drive the slides. Use **minimal text**.
4. **One key message per slide** — each slide must have a clear headline summarizing its takeaway.
5. **Professional template** — clean, consistent, and readable. All text must be large enough to read from the back of the room.
6. **Tasks 4.1, 4.2, 4.3 are NOT presented** — they are covered exclusively during viva Q&A. Every member must understand them.
7. **Every member must know the entire project** — the examiner can ask any member about any phase.
8. **Every member must understand the code they wrote** — even though code is not shown in slides, the examiner will ask about it during viva.

### 6-Member Task Split (Per Guidelines)

| Presenter | Task                                        |
|-----------|---------------------------------------------|
| Presenter 1 | Task 3.1 — EDA & Preprocessing Pipeline  |
| Presenter 2 | Task 3.2 — Continuous Price Prediction    |
| Presenter 3 | Task 3.3A — Customer Segment Classification |
| Presenter 4 | Task 3.3B — Delivery Status Classification  |
| Presenter 5 | Task 3.4 — Performance Analysis & Risk    |
| Presenter 6 | Task 3.5 — Unsupervised Clustering        |

---

## 2. TEAM ASSIGNMENT & TIME ALLOCATION

| #  | Presenter                           | Student ID  | Task                              | Time   | Slides       |
|----|-------------------------------------|-------------|-----------------------------------|--------|--------------|
| 1  | **Dilusha Sandaruwan** (Leader)     | 2301691099  | Task 3.1 — EDA & Preprocessing    | 5 min  | Slides 1–5   |
| 2  | **Madusha Lakmina**                 | 2301691027  | Task 3.2 — Price Prediction       | 6 min  | Slides 6–9   |
| 3  | **Harsha Dilan**                    | 2301691017  | Task 3.3A — Customer Segments     | 5 min  | Slides 10–12 |
| 4  | **Thanura Vipulanga**               | 2301691013  | Task 3.3B — Delivery Status       | 4 min  | Slides 13–14 |
| 5  | **Chamod Thejan**                   | 2301691108  | Task 3.4 — Performance & Risk     | 5 min  | Slides 15–17 |
| 6  | **Dasun Wijayathilaka**             | 2301691094  | Task 3.5 — Clustering + Closing   | 5 min  | Slides 18–21 |
|    |                                     |             | **TOTAL**                         | **30 min** | **21 slides** |

> **Note:** Average pace ≈ 1.4 minutes per slide. This is ideal for a visual-heavy, minimal-text deck.

---

## 3. SLIDE-BY-SLIDE BREAKDOWN

---

### PRESENTER 1 — Dilusha Sandaruwan Karunathilaka (Leader)

**Task:** 3.1 — Exploratory Data Analysis & Preprocessing Pipeline  
**Time:** 5 minutes | **Slides:** 1–5  
**Notebook Reference:** `notebooks/1_eda_and_preprocessing.ipynb`

---

#### SLIDE 1 — Title & Team Introduction

**Headline:** *AuraCart Sentinel — Intelligent Retail Analytics & MLOps System*

**Visual Layout:**
- "AuraCart Sentinel" branding with a modern, professional design
- CoreGenix team name and group identifier (GDSE 69)
- Photos or names of all 6 members with their assigned tasks
- Module: ITS 2140 — Machine Learning
- Date: April 2026

**Key Message:** Establish the team identity and project scope.

**Speech:**

> "Good morning, everyone. We are Team CoreGenix from GDSE 69, Panadura. Today, we are proud to present 'AuraCart Sentinel' — an end-to-end machine learning system built to transform AuraCart from a traditional e-commerce platform into an intelligent, data-driven enterprise.
>
> As the team leader, I coordinated our efforts across all project phases — from exploratory analysis through to cloud deployment. Allow me to introduce our team: I am Dilusha Sandaruwan, handling EDA and preprocessing. Madusha covers price prediction, Harsha handles customer segmentation, Thanura takes on delivery status classification, Chamod leads our performance analysis, and Dasun presents the unsupervised clustering work.
>
> Let me begin with the foundation of everything — our data exploration and preprocessing pipeline."

---

#### SLIDE 2 — The Business Problem & Project Objectives

**Headline:** *Three Critical Challenges, One Sentinel System*

**Visual Layout:**
- Three large icons or columns representing:
  1. 💰 **Revenue Uncertainty** — "How much will this order cost?"
  2. 👥 **Customer Blind-spots** — "Who are our VIP customers?"
  3. 🚚 **Logistics Risk** — "Will this order be delivered on time?"
- Below: A pipeline diagram showing the three ML paradigms: **Regression → Classification → Clustering**
- Arrow pointing to: **Cloud Deployment (Vertex AI)**

**Key Message:** AuraCart needs predictive intelligence across three simultaneous domains.

**Speech:**

> "AuraCart is a rapidly scaling, mid-tier e-commerce platform dealing with over 10,000 transactions. They face three operational challenges. First, they cannot accurately forecast revenue from incoming orders. Second, they have no automated way to identify who their VIP customers are versus new or returning shoppers. Third, they struggle to predict logistics outcomes — will an order be delivered, delayed, or returned?
>
> Sentinel was designed to solve all three using three distinct ML paradigms: regression for price prediction, multi-class classification for customer segments and delivery status, and unsupervised clustering to discover hidden customer behaviours. The final system is tracked through MLflow and deployed live on Google Cloud Vertex AI."

---

#### SLIDE 3 — Dataset Overview & Key Challenges

**Headline:** *10,000 Orders — Real Challenges in Synthetic Data*

**Visual Layout:**
- Dataset summary card: Source (Hugging Face — `millat/e-commerce-orders`), Records (10,000), Features (15 original → 20 after engineering)
- Two mini bar charts showing class imbalance:
  - **Customer Segment:** New 50% | Returning 35% | VIP 15%
  - **Delivery Status:** Delivered 70% | Shipped 20% | Pending 5% | Returned 5%
- Target variables highlighted: `price` (continuous), `customer_segment` (3-class), `delivery_status` (4-class)

**Key Message:** The dataset presents real-world challenges — class imbalance and weak pricing signals.

**Speech:**

> "Our dataset comes from Hugging Face and contains 10,000 e-commerce transactions with 15 features. After feature engineering — including datetime decomposition and calculating days-to-ship — we expanded this to 20 modelling features.
>
> The most critical challenge is class imbalance. For customer segments, VIP customers make up only 15% of the data — this is the group AuraCart cares about most. For delivery status, 'Returned' orders are just 5%, yet these are the most expensive for the business. Price follows a roughly uniform distribution from $5 to $500 with weak linear signals to other features. These characteristics directly shaped our modelling strategy."

---

#### SLIDE 4 — EDA Key Findings

**Headline:** *What the Data Revealed*

**Visual Layout:**
- **Left column:** Continuous Feature Distributions histogram (`reports/figures/continuous_distributions.png`)
- **Right column:** Pearson Correlation Heatmap (`reports/figures/correlation_heatmap.png`)
- **Bottom strip:** Categorical Feature Distribution (`reports/figures/categorical_distributions.png`) — smaller thumbnail

**Key Message:** Weak feature correlations confirmed the need for advanced feature engineering.

**Speech:**

> "Our EDA revealed several key patterns. Price follows a nearly uniform distribution — there's no natural clustering around specific price points. Quantity is uniformly distributed between 1 and 5 units. Temporal features like order hour and day of the week showed minimal seasonal patterns.
>
> The correlation heatmap confirmed something critical: most feature pairs have weak linear correlations. This told us that a simple linear model would underfit the data for price prediction, and for classification, we would need to engineer stronger behavioural features. Category, payment method, device type, and sales channel all have approximately uniform distributions across their values."

---

#### SLIDE 5 — Preprocessing Pipeline Architecture

**Headline:** *From Raw Data to Production-Ready Features*

**Visual Layout:**
- A pipeline flow diagram showing:
  - **Input:** Raw 15-feature dataset
  - **Step 1 — Feature Engineering:** Datetime decomposition → `order_date_month`, `order_date_day`, `order_date_hour`, `days_to_ship`
  - **Step 2 — ColumnTransformer:**
    - *Numerical path:* StandardScaler → 17+ features (zero-mean, unit-variance)
    - *Categorical path:* OneHotEncoder → `category`, `payment_method`, `device_type`, `channel` (with `handle_unknown='ignore'`)
  - **Output:** 20+ clean features → `preprocessing_pipeline.joblib`
- Small icon showing the serialized `.joblib` file

**Key Message:** A reproducible, serialized pipeline ensures consistent preprocessing from training to production.

**Speech:**

> "A model is only as good as its input data. We built a robust Scikit-learn preprocessing pipeline using ColumnTransformer. Numerical features go through StandardScaler for zero-mean, unit-variance normalization. Categorical features — category, payment method, device type, and channel — are encoded using OneHotEncoder with unknown-category handling enabled.
>
> Critically, this entire pipeline  is serialized as a `.joblib` file. This means the exact same transformations applied during training are automatically applied during inference in production. Whether we're training locally or predicting on Vertex AI, the preprocessing is identical and reproducible. This pipeline outputs 20+ clean features ready for modelling.
>
> With our data foundation established, I'll hand over to Madusha to present how we used it for price prediction."

---

### PRESENTER 2 — Madusha Lakmina

**Task:** 3.2 — Continuous Price Prediction Modeling  
**Time:** 6 minutes | **Slides:** 6–9  
**Notebook Reference:** `notebooks/2_supervised_modeling.ipynb` (Regression section)

---

#### SLIDE 6 — Regression Approach: OLS vs SGD

**Headline:** *Predicting Order Price — The First Layer of Sentinel*

**Visual Layout:**
- Two-panel comparison:
  - **Left panel:** OLS Linear Regression (closed-form solution — one-shot calculation)
  - **Right panel:** SGD Regressor (iterative gradient descent — learns step-by-step)
- Diagram showing the concept: Input features → Linear combination → Predicted price ($5–$500)
- Small callout: "Goal: Predict the `price` of each order for revenue forecasting"

**Key Message:** We tested both analytical (OLS) and iterative (SGD) approaches to understand regression behaviour.

**Speech:**

> "Thank you, Dilusha. The first analytical layer of AuraCart Sentinel is revenue prediction — forecasting the monetary value of incoming orders.
>
> We implemented two approaches. First, Ordinary Least Squares — known as OLS — which computes the optimal weights in a single mathematical step using the normal equation. Second, Stochastic Gradient Descent, or SGD, which iteratively adjusts weights by descending the error surface step-by-step.
>
> The purpose of testing both wasn't just to find the best model — it was to understand how hyperparameters like learning rate and training epochs affect convergence, which is a core requirement of this project."

---

#### SLIDE 7 — SGD Hyperparameter Experimentation

**Headline:** *Learning Rate and Epoch Impact — 5 SGD Configurations*

**Visual Layout:**
- Results table (large, readable):

| Config   | Learning Rate | Epochs | Test MSE        | Status      |
|----------|:------------:|:------:|----------------:|:-----------:|
| SGD-1    | 0.001        | 100    | 14,925.82       | ✅ Converged |
| SGD-2    | 0.01         | 100    | 15,883.78       | ✅ Converged |
| SGD-3    | 0.001        | 500    | 15,012.90       | ✅ Converged |
| SGD-4    | 0.01         | 500    | 16,639.47       | ✅ Converged |
| SGD-5    | 0.1          | 1000   | **380,541,523+** | ❌ Diverged  |

- Visual highlight on Config 5 showing divergence (red color/warning icon)
- Small bar chart comparing the MSE values (excluding Config 5)

**Key Message:** Learning rate 0.1 caused catastrophic divergence — the optimal range is 0.001–0.01.

**Speech:**

> "We tested five distinct SGD configurations to systematically explore the effect of learning rate and training epochs. Configs 1 through 4 all converged successfully, with learning rate 0.001 at 100 epochs producing the lowest test MSE of approximately 14,926.
>
> The critical observation was Config 5 — learning rate 0.1 with 1,000 epochs. This caused the model to completely diverge, producing an MSE in the hundreds of millions. This perfectly demonstrates why learning rate selection is so critical in gradient descent optimization. Too large a step overshoots the minimum and the loss explodes.
>
> The optimal range for this dataset is 0.001 to 0.01, and longer training epochs provided diminishing returns — indicating the model converges relatively quickly."

---

#### SLIDE 8 — Cross-Validation & Bias-Variance Analysis

**Headline:** *5-Fold Cross-Validation — Stable but Underfitting*

**Visual Layout:**
- Results card (prominent):

| Metric         | Value                | Assessment                     |
|---------------|---------------------|---------------------------------|
| Test MAE      | **$98.94**          | ~$99 average per-order error    |
| Test RMSE     | **$122.23**         | Interpretable error magnitude   |
| CV MAE (mean) | **$97.52 (±$0.89)** | Extremely stable across folds   |
| CV MSE (mean) | **14,487.70 (±275.57)** | Low variance                |

- Bias-Variance diagram:
  - Train MSE ≈ CV MSE → **High Bias (Underfitting)**
  - Arrow explaining: "Model is too simple for the data — expected for synthetic pricing data"
- Optional: Bar chart showing per-fold CV results: [14,940 | 14,255 | 14,678 | 14,291 | 14,275]

**Key Message:** The model is stable (CV MAE ±$0.89) but underfits — a realistic outcome for synthetic data.

**Speech:**

> "To rigorously validate our model, we applied 5-fold cross-validation. The results show exceptional stability — our CV MAE is $97.52 with a standard deviation of only $0.89. This means the model performs consistently regardless of which data split it sees.
>
> However, notice that the training MSE and cross-validation MSE are nearly identical. This is a clear indicator of underfitting — the model has high bias. The linear model is too simple to capture the relationships in this data.
>
> This is actually an expected and realistic outcome. The pricing data in this dataset is synthetically generated without strong signal features. Real-world price prediction would involve stronger correlations with factors like product cost and season. For AuraCart's purposes, a ±$99 error on orders ranging from $5 to $500 provides usable baseline accuracy for inventory planning."

---

#### SLIDE 9 — Regression Business Impact

**Headline:** *What Does a $99 Error Mean for AuraCart?*

**Visual Layout:**
- Business impact infographic:
  - 📦 **Per-order:** ±$99 average prediction error on a $5–$500 range
  - 📊 **Monthly forecasting:** With 10,000 orders/month, cumulative errors average out significantly
  - 💡 **Actionable:** Sufficient for inventory pre-allocation and cash flow planning
- A "What-if" scenario box:
  - "If AuraCart receives 500 orders tomorrow → total revenue can be estimated within ±5% aggregate error"
- SGD takeaway callout: "SGD showed us that hyperparameter sensitivity matters — lr=0.1 destroyed the model"

**Key Message:** Even with underfitting, the model provides AuraCart with usable revenue forecasting for operations.

**Speech:**

> "So what does a $99 average error actually mean for AuraCart's business? On individual orders — where the price ranges from $5 to $500 — this error is moderate. But at scale, with thousands of daily transactions, individual errors average out.
>
> For AuraCart's use case — monthly revenue forecasting and inventory pre-allocation — this precision is operationally useful. They don't need exact per-order pricing; they need reliable aggregate estimates for supply chain planning.
>
> The SGD experimentation also provided a critical engineering lesson: hyperparameter sensitivity is real. A single misconfigured learning rate can destroy model performance entirely. This reinforced the importance of the systematic experiment tracking we implemented through MLflow.
>
> Now I'll hand over to Harsha, who will present how we classified AuraCart's customers into meaningful segments."

---

### PRESENTER 3 — Harsha Dilan

**Task:** 3.3A — Multi-Class Classification: Customer Segment Prediction  
**Time:** 5 minutes | **Slides:** 10–12  
**Notebook Reference:** `notebooks/2_supervised_modeling.ipynb` (Customer Segment Classification section)

---

#### SLIDE 10 — Customer Segment: Problem & Feature Engineering

**Headline:** *Not All Customers Are Equal — Engineering Behavioural Features*

**Visual Layout:**
- **Top section:** Pie chart or bar chart showing class distribution:
  - New: 50% | Returning: 35% | VIP: 15%
- **Bottom section:** Feature engineering diagram:
  - Raw order-level data → **Aggregation per customer** → 6 behavioural features:
    1. `order_frequency` — Number of orders per customer
    2. `avg_spend` — Average order value
    3. `total_spend` — Lifetime revenue
    4. `avg_quantity` — Average items per order
    5. `product_diversity` — Unique products purchased
    6. `category_diversity` — Unique categories explored
  - **Leakage prevention callout:** "Training data ONLY → Test customers get median fallback (9.4% unseen)"

**Key Message:** Behavioural feature engineering, done without leakage, was the breakthrough for classification.

**Speech:**

> "Thank you, Madusha. The second core task is customer segmentation — predicting whether a customer is New, Returning, or VIP based on their purchasing behaviour.
>
> The challenge is clear from the distribution: VIP customers represent only 15% of the data. These are the customers AuraCart cares about most, but they're the hardest to detect.
>
> The breakthrough came from feature engineering. Instead of using raw order-level features, we aggregated transaction history per customer to create six behavioural features — order frequency, average and total spend, average quantity, and product and category diversity. These features capture the *patterns* that distinguish a loyal VIP from a first-time shopper.
>
> Critically, to prevent data leakage, all aggregations are computed exclusively from training data. The 9.4% of test customers who were never seen during training receive median fallback values. Without this safeguard, accuracy was artificially inflated — we caught and corrected this during development."

---

#### SLIDE 11 — Model Comparison & Champion Selection

**Headline:** *Gradient Boosting Wins — 81.80% Accuracy*

**Visual Layout:**
- **Model comparison bar chart:**

| Model               | Accuracy |
|---------------------|:--------:|
| Logistic Regression | 75.00%   |
| Random Forest       | 67.65%   |
| **Gradient Boosting** | **81.80%** ✅ |

- **Champion model card:**
  - Algorithm: Gradient Boosting Classifier
  - `n_estimators=300`, `max_depth=5`, `learning_rate=0.1`
  - Weighted F1: 0.8165
  - Log-Loss: 0.5742
- Small note: "Also explored: LightGBM, Ensemble Voting, Cost-Sensitive Learning — GBM was most balanced"

**Key Message:** After systematic evaluation of 15+ MLflow-tracked experiments, Gradient Boosting achieved the best balance.

**Speech:**

> "We evaluated multiple classifier architectures systematically. Logistic Regression provided a solid baseline at 75%, but couldn't capture the non-linear decision boundaries needed. Surprisingly, Random Forest underperformed at 67.65%, likely due to the synthetic feature distributions.
>
> Our champion model is Gradient Boosting with 300 estimators, max depth of 5, and a learning rate of 0.1. It achieved 81.80% accuracy — the highest among all tested approaches — with a weighted F1-score of 0.8165 and a log-loss of 0.5742.
>
> We also explored LightGBM, soft and hard ensemble voting, and cost-sensitive learning with VIP penalty weighting. Over 15 experiment runs were tracked in MLflow, and the Gradient Boosting model was selected as the champion based on balanced per-class performance."

---

#### SLIDE 12 — Customer Segment: Final Results

**Headline:** *Segment Results — Where the Model Excels and Where It Struggles*

**Visual Layout:**
- **Confusion Matrix** (main visual): `reports/figures/cm_customer_segment.png`
- **Quick metrics sidebar:**
  - Accuracy: 81.80%
  - F1-Score (Weighted): 0.8165
  - Log-Loss: 0.5742
- **Business takeaway callout:**
  - "✅ VIP detection enables automated loyalty targeting"
  - "✅ Segment-driven marketing can begin with 80%+ confidence"

**Key Message:** 81.80% accuracy with balanced per-class performance makes this model production-ready for marketing automation.

**Speech:**

> "Here's our confusion matrix for customer segment prediction. At 81.80% accuracy, the model provides reliable segmentation across all three categories. The confusion matrix reveals exactly where misclassifications occur — primarily between New and Returning customers, which is expected since their behavioural profiles overlap the most.
>
> What matters most for AuraCart is VIP detection. The model successfully identifies the majority of high-value customers, enabling automated loyalty programs, personalised marketing campaigns, and targeted retention strategies with over 80% confidence.
>
> This model was selected as our champion for deployment to Google Cloud Vertex AI — the one that serves live predictions today.
>
> Now Thanura will present the second classification challenge — delivery status prediction."

---

### PRESENTER 4 — Thanura Vipulanga

**Task:** 3.3B — Multi-Class Classification: Delivery Status Prediction  
**Time:** 4 minutes | **Slides:** 13–14  
**Notebook Reference:** `notebooks/2_supervised_modeling.ipynb` (Delivery Status Classification section)

---

#### SLIDE 13 — Delivery Status: The Class Imbalance Challenge

**Headline:** *Predicting Logistics Outcomes — 70% Majority Class Dominance*

**Visual Layout:**
- **Bar chart showing extreme imbalance:**
  - Delivered: **70%** (massive dominant bar)
  - Shipped: **20%**
  - Pending: **5%** (tiny bar)
  - Returned: **5%** (tiny bar — highlighted as "Most Costly")
- **Cost annotation:**
  - Returns cost AuraCart the most: refunds + reverse logistics + lost inventory
  - Delayed/Pending orders erode customer trust
- Callout: "The minority classes are the ones that matter most to the business"

**Key Message:** Delivery status prediction is the hardest task due to extreme class imbalance — the rare events are the expensive ones.

**Speech:**

> "Thank you, Harsha. I'm presenting the second classification task — delivery status prediction. The goal is to predict whether an order will be Delivered, Shipped, Pending, or Returned.
>
> The fundamental challenge here is the extreme class imbalance. 70% of all orders are in the 'Delivered' category — the model can achieve 70% accuracy by simply predicting 'Delivered' for everything. But the rare classes — Pending and Returned at just 5% each — are precisely the ones that cost AuraCart real money.
>
> Returns involve refund processing, reverse logistics, and lost inventory. Yet they represent only 5% of the training data, making them extremely hard for the model to learn. This is the classic imbalanced classification problem."

---

#### SLIDE 14 — Delivery Status: Results & Threshold Analysis

**Headline:** *70.45% Accuracy — Baseline with Majority-Class Learning*

**Visual Layout:**
- **Confusion Matrix** (main visual): `reports/figures/cm_delivery_status.png`
- **Metrics card:**
  - Accuracy: **70.45%**
  - Weighted F1: **0.5894**
  - Model: Multinomial Logistic Regression (baseline)
- **Threshold Calibration mini-table:**
  - Threshold sweep for "Returned" class: 0.10 → 0.50
  - Lower threshold → more returns caught (higher recall) but more false alarms (lower precision)
- **Honest assessment callout:**
  - "Majority-class learning captures Delivered well"
  - "Minority classes (Pending, Returned) remain challenging"
  - "Future: SMOTE, cost-weighted penalties, ensemble methods"

**Key Message:** 70.45% accuracy reflects natural majority-class learning. Minority detection requires advanced imbalance techniques.

**Speech:**

> "Our delivery status model achieves 70.45% accuracy using Multinomial Logistic Regression. The weighted F1-score is 0.5894. Looking at the confusion matrix, the model correctly predicts 'Delivered' orders well — which makes up 70% of all data. However, the minority classes — Pending and Returned — are significantly harder to detect.
>
> To address this, we performed threshold calibration on the 'Returned' class, sweeping thresholds from 0.10 to 0.50. Lowering the threshold increases recall — we catch more returns — but at the cost of more false alarms. This precision-recall trade-off is central to the business decision AuraCart needs to make: is it worth flagging more orders as potential returns if it means some false positives?
>
> For future improvement, we recommend applying SMOTE for synthetic oversampling, cost-weighted objective functions, and ensemble methods. These strategies could significantly boost minority-class detection.
>
> I'll now hand over to Chamod, who will dive deep into how we evaluated and compared our model performance."

---

### PRESENTER 5 — Chamod Thejan

**Task:** 3.4 — Classification Performance Analysis & Risk Evaluation  
**Time:** 5 minutes | **Slides:** 15–17  
**Notebook Reference:** `notebooks/2_supervised_modeling.ipynb` (Performance Analysis section)

---

#### SLIDE 15 — Confusion Matrix Deep-Dive

**Headline:** *Reading the Confusion Matrix — Where Models Succeed and Fail*

**Visual Layout:**
- **Side-by-side confusion matrices:**
  - Left: Customer Segment CM (`cm_customer_segment.png`)
  - Right: Delivery Status CM (`cm_delivery_status.png`)
- **Annotated insights:**
  - Customer Segment: "Most confusion between New ↔ Returning (overlapping behaviour)"
  - Delivery Status: "Model predicts Delivered for most entries — minority classes under-detected"
- **Class-wise metrics table for Customer Segment (visible):**

| Class      | Precision | Recall | F1-Score |
|------------|:---------:|:------:|:--------:|
| New        | —         | —      | —        |
| Returning  | —         | —      | —        |
| VIP        | —         | —      | —        |

*(Values from the actual classification report output in notebooks)*

**Key Message:** The confusion matrix reveals misclassification *patterns*, not just a single accuracy number.

**Speech:**

> "Thank you, Thanura. My task was to rigorously evaluate model performance and analyse the risks of misclassification.
>
> The confusion matrix is our most powerful diagnostic tool. For customer segments, you can see the model performs strongly overall — but the errors concentrate in the New-versus-Returning boundary. These two segments have the most behavioural overlap, which makes them genuinely hard to separate.
>
> For delivery status, the picture is different. The model has learned to predict 'Delivered' with high confidence — because that's 70% of the data. But the critical minority classes — Pending and Returned — are frequently misclassified as Delivered. This isn't just a modelling issue; it's a business risk. Each missed return is a missed opportunity for proactive customer service."

---

#### SLIDE 16 — Precision-Recall Trade-offs

**Headline:** *The Precision-Recall Trade-off — You Can't Have Both*

**Visual Layout:**
- **Threshold sweep results for "Returned" class:**

| Threshold | Precision | Recall | F1-Score |
|:---------:|:---------:|:------:|:--------:|
| 0.10      | Lower     | Higher | —        |
| 0.20      | —         | —      | —        |
| 0.30      | —         | —      | —        |
| 0.40      | —         | —      | —        |
| 0.50      | Higher    | Lower  | —        |

*(Populate with exact values from notebook output)*

- **Conceptual diagram:**
  - Sliding scale: ← More Recalls (catch more returns) ←→ More Precision (fewer false alarms) →
  - Business question: "Is it better to flag too many orders or miss actual returns?"

**Key Message:** Lowering the threshold catches more returns but creates more false alarms — the business must choose.

**Speech:**

> "A critical concept in classification is the precision-recall trade-off. Precision asks: 'Of all orders flagged as returns, how many actually are?' Recall asks: 'Of all actual returns, how many did we catch?'
>
> We performed a threshold sweep for the Returned class from 0.10 to 0.50. At a low threshold of 0.10, we catch more actual returns — higher recall — but many non-returns get flagged incorrectly, creating more work for the logistics team. At a high threshold of 0.50, we're very precise — almost every flagged order is genuinely at risk — but we miss many actual returns.
>
> For AuraCart, we recommend a lower threshold. The cost of missing a return — processing refunds after the fact, dealing with customer complaints — is higher than the cost of proactively reaching out to a customer whose order was actually fine. Being 'safe' is more valuable than being 'perfectly precise' in this context."

---

#### SLIDE 17 — Asymmetric Risk: The 5x VIP Penalty Strategy

**Headline:** *Not All Mistakes Are Equal — The Cost of Being Wrong*

**Visual Layout:**
- **Cost comparison table:**

| Error Type | Example | Business Cost |
|------------|---------|:------------:|
| **False Positive (VIP)** | Regular customer treated as VIP | Low — minor over-service |
| **False Negative (VIP)** | VIP customer treated as New | **HIGH** — lost loyalty, churn risk |
| **False Positive (Return)** | Good order flagged as risky | Moderate — extra verification |
| **False Negative (Return)** | Returned order missed | **HIGH** — lost revenue + logistics |

- **5x Penalty Strategy callout:**
  - "We penalized VIP misclassification at 5x the standard rate"
  - "This shifted the model's focus: protect VIPs at all costs"
- **Business principle:** "Tolerate false positives to eliminate false negatives on high-value classes"

**Key Message:** Asymmetric risk means the model should prioritize protecting VIPs and catching returns, even at the cost of some false alarms.

**Speech:**

> "The final critical insight from our performance analysis is that not all misclassifications carry equal cost. In the real world, errors have asymmetric consequences.
>
> Missing a VIP customer — treating them like a new buyer — means no loyalty reward, no personalized service, and a risk of losing their business entirely. But accidentally treating a regular customer as a VIP? That's just a minor extra discount — a small cost with potential upside.
>
> The same logic applies to delivery: missing an actual return is far more expensive than preemptively flagging a safe order for review.
>
> To formalize this, we implemented cost-sensitive learning with a 5x penalty for VIP misclassification. We told the model: 'It is five times worse to miss a VIP than to over-promote a regular customer.' This approach is grounded in real business risk analysis, not just mathematical optimization.
>
> Now I'll hand over to Dasun for the final task — discovering hidden patterns through unsupervised clustering."

---

### PRESENTER 6 — Dasun Wijayathilaka

**Task:** 3.5 — Unsupervised Learning: Customer Behaviour Clustering + Project Conclusion  
**Time:** 5 minutes | **Slides:** 18–21  
**Notebook Reference:** `notebooks/3_unsupervised_clustering.ipynb`

---

#### SLIDE 18 — Clustering Methodology & Optimal k

**Headline:** *Letting the Data Speak — K-Means Clustering*

**Visual Layout:**
- **Elbow Method + Silhouette Score plot** (main visual): `reports/figures/elbow_silhouette.png`
- **Method explanation callout:**
  - Input: Scaled numerical features (quantity, price, days_to_ship, temporal features)
  - Algorithm: K-Means (iterative centroid optimization)
  - Selection method: Elbow point at **k=4** confirmed by silhouette scores
- **Highlight:** Arrow pointing to the elbow bend on the WCSS curve

**Key Message:** The Elbow Method and Silhouette Scores both confirm k=4 as the optimal number of customer clusters.

**Speech:**

> "Thank you, Chamod. For the final task, we shifted from supervised learning to unsupervised discovery. Instead of predicting known labels, we let the data reveal hidden customer behaviour patterns using K-Means clustering.
>
> We applied K-Means to our scaled numerical features — including quantity, price, days to ship, and temporal features. To determine the optimal number of clusters, we used two complementary techniques: the Elbow Method, which plots Within-Cluster Sum of Squares against k, and Silhouette Scores, which measure how well each data point fits its assigned cluster.
>
> Both methods clearly identify k=4 as the optimal number. You can see the distinct 'elbow' bend on the WCSS curve at this point. Beyond k=4, adding more clusters provides diminishing separation."

---

#### SLIDE 19 — Cluster Profiles & PCA Visualization

**Headline:** *Four Hidden Customer Segments Discovered*

**Visual Layout:**
- **PCA 2D Cluster Visualization** (main visual): `reports/figures/cluster_pca.png`
- **Cluster summary table:**

| Cluster | Size   | Key Characteristics              | Business Label                |
|---------|--------|----------------------------------|-------------------------------|
| 0       | ~2,500 | High days_to_ship, moderate qty  | **Slow-Fulfilment Orders**    |
| 1       | ~2,000 | Low price, high frequency        | **Budget-Conscious Buyers**   |
| 2       | ~2,500 | High price, high total spend     | **Premium / VIP Customers**   |
| 3       | ~2,000 | Moderate across all features     | **Average / Returning**       |

- Optional smaller visual: `reports/figures/cluster_boxplots.png` (feature distributions per cluster)

**Key Message:** K-Means discovered four meaningful customer behavioural groups without using any labels.

**Speech:**

> "The PCA visualization projects our four clusters into 2D space. You can see reasonably distinct groupings — the clusters are not perfectly separated, which is realistic for e-commerce data, but they capture genuinely different behaviours.
>
> Cluster 0 represents slow-fulfilment orders — customers who experience longer shipping times. Cluster 1 captures budget-conscious, frequent buyers who place smaller orders but shop often. Cluster 2 is our premium segment — high price, high total spend. And Cluster 3 represents the moderate middle — average behaviour across all features.
>
> What's particularly interesting is that when we cross-tabulated these unsupervised clusters with the known customer segment labels, we found meaningful overlap. The algorithm independently discovered segments that correspond to our labelled categories — validating both approaches."

---

#### SLIDE 20 — Business Insights from Clustering

**Headline:** *From Clusters to Action — Personalised Strategies*

**Visual Layout:**
- **Four action cards** (one per cluster):

| Cluster | Business Label           | Recommended Action                              |
|---------|--------------------------|------------------------------------------------|
| 0       | Slow-Fulfilment          | 🚚 Logistics optimization, offer express options |
| 1       | Budget-Conscious         | 💰 Discount promotions, bundle deals            |
| 2       | Premium / VIP            | ⭐ Premium service, loyalty rewards, priority shipping |
| 3       | Average / Returning      | 🔄 Retention campaigns, upsell opportunities    |

- **Cross-functional impact diagram:**
  - Marketing → Targeted campaigns per cluster
  - Operations → Shipping tier optimization
  - Finance → Dynamic pricing strategies
  - Customer Success → Personalised engagement

**Key Message:** Each cluster maps to a concrete business action — this is actionable intelligence, not just data science.

**Speech:**

> "The true value of clustering lies in what AuraCart can *do* with these segments. Each cluster maps to a specific business strategy.
>
> For Cluster 0 — the slow-fulfilment group — AuraCart can offer express shipping upgrades or optimize their logistics pipeline for this cohort. Interestingly, some of these customers may not mind slower shipping if the price is right — opening the door for a new economy shipping tier.
>
> Cluster 1 — budget-conscious buyers — responds best to discount promotions and bundle deals. Cluster 2 — the premium segment — warrants priority shipping, loyalty rewards, and premium customer service. And Cluster 3 — the average segment — represents the largest opportunity for retention and upselling campaigns.
>
> This maps not just to marketing, but across operations, finance, and customer success — making the clustering analysis genuinely cross-functional."

---

#### SLIDE 21 — Summary, Conclusion & Thank You

**Headline:** *AuraCart Sentinel — Intelligence On Duty*

**Visual Layout:**
- **Final Performance Scorecard:**

| Task                 | Model              | Key Metric  | Result   |
|----------------------|--------------------|-----------:|:--------:|
| Price Prediction     | Linear Regression  | MAE         | $98.94   |
| Customer Segments    | Gradient Boosting  | Accuracy    | 81.80%   |
| Delivery Status      | Logistic Regression| Accuracy    | 70.45%   |
| Customer Clustering  | K-Means (k=4)     | Clusters    | 4 found  |

- **Production Pipeline Summary:**
  - ✅ Tracked in MLflow (20+ experiments)
  - ✅ Packaged as `.joblib` pipeline
  - ✅ Deployed to Google Cloud Vertex AI
  - ✅ Live predictions via REST API
- **"Thank You" and team logo/names**

**Key Message:** AuraCart Sentinel delivers a complete, production-grade ML system from data exploration to cloud deployment.

**Speech:**

> "To conclude, AuraCart Sentinel delivers intelligence across three domains: price prediction with a $99 average error, customer segmentation at 81.80% accuracy, delivery status monitoring at 70.45%, and discovery of four hidden customer clusters — all within a unified system.
>
> Importantly, this isn't a notebook experiment. Every model run is tracked in MLflow with full reproducibility. The champion model is serialized as a production pipeline and deployed to Google Cloud Vertex AI, serving live predictions via REST API. AuraCart can send real-time order data to our cloud endpoint and receive instant customer segment predictions.
>
> Sentinel is officially on duty. We are Team CoreGenix, and we are ready for your questions. Thank you."

---

## 4. VIVA VOCE PREPARATION GUIDE

> **Critical:** The examiner will ask each member questions for ~5 minutes — both about their own section AND about other parts of the project. Tasks 4.1, 4.2, and 4.3 are **only assessed during viva**, not in presentation slides.

### 4.1 — Universal Questions (Every Member Must Know)

| Topic | Key Facts to Know |
|-------|-------------------|
| **Dataset** | 10,000 records, Hugging Face, 15 features → 20 after engineering |
| **Class Imbalance** | Customer: 50/35/15; Delivery: 70/20/5/5 |
| **Preprocessing** | ColumnTransformer, StandardScaler, OneHotEncoder, handle_unknown='ignore' |
| **Why not Linear Regression for classification?** | Outputs are unbounded, not probabilities. Softmax converts logits to probabilities. |
| **Softmax Function** | P(y=k\|x) = exp(wk·x) / Σ exp(wj·x) — converts raw scores to probabilities that sum to 1 |
| **Cross-Entropy Loss** | -Σ y_true · log(y_pred) — penalizes confident wrong predictions heavily |
| **Train-Test Split** | 80/20 stratified split to maintain class proportions |
| **Random Seed** | random_state=42 used throughout for reproducibility |

### 4.2 — Task 4.1: MLflow Experiment Tracking (Viva Only)

**What you must explain:**

1. **What is MLflow?** — Open-source experiment tracking platform that logs parameters, metrics, and artifacts for ML runs.
2. **How was it integrated?**
   - `mlflow.start_run()` wraps each experiment
   - `mlflow.log_param()` logs hyperparameters
   - `mlflow.log_metric()` logs accuracy, F1, MSE, etc.
   - `mlflow.sklearn.log_model()` saves the model artifact
3. **Experiments created:**
   - `AuraCart_Regression` — 6 runs (OLS + 5 SGD)
   - `AuraCart_Classification` — 15+ runs (LR, RF, GBM, LightGBM, ensemble, cost-sensitive)
   - `AuraCart_Deployment` — Final champion model
4. **Why MLflow matters:**
   - Reproducibility: every run is recorded with all parameters and environment
   - Comparison: side-by-side metric comparison across runs
   - Auditability: can trace back to exact hyperparameters for any result
5. **Screenshots to reference:** `screenshots/mlflow/` folder

### 4.3 — Task 4.2: Model Packaging & Artifact Management (Viva Only)

**What you must explain:**

1. **Champion model selected:** Gradient Boosting Classifier for customer segments (81.80% accuracy)
2. **Unified Pipeline:**
   ```
   Pipeline([
     ('preprocessor', ColumnTransformer([
       ('num', StandardScaler(), [numerical features]),
       ('cat', OneHotEncoder(), [categorical features])
     ])),
     ('classifier', GradientBoostingClassifier(...))
   ])
   ```
3. **Serialization:** `joblib.dump(pipeline, 'model.joblib')` — saves entire pipeline including preprocessing
4. **Artifacts produced:**
   - `artifacts/model.joblib` — deployed model pipeline
   - `artifacts/preprocessing_pipeline.joblib` — standalone preprocessing
   - `artifacts/requirements.txt` — Python dependencies
5. **Why a Pipeline?** Raw features go in → predictions come out. No separate preprocessing step needed at inference time.

### 4.4 — Task 4.3: Vertex AI Deployment (Viva Only)

**What you must explain:**

1. **Deployment Steps:**
   - Upload artifacts to GCS bucket (`auracart-sentinel-ml01-artifacts`)
   - Register model in Vertex AI Model Registry using Scikit-learn pre-built container
   - Create endpoint and deploy model (`n1-standard-2`, 100% traffic)
2. **gcloud commands used:**
   ```
   gcloud ai models upload --region=us-central1 --display-name=auracart-segment-classifier ...
   gcloud ai endpoints create --display-name=auracart-deployment ...
   gcloud ai endpoints deploy-model ENDPOINT_ID --model=MODEL_ID ...
   ```
3. **Test prediction payload:**
   ```json
   {"instances": [{"category": "Electronics", "quantity": 3, "payment_method": "Credit Card", ...}]}
   ```
4. **What happens when a prediction is requested:**
   - JSON payload → Vertex AI endpoint → deserializes model.joblib → Pipeline preprocesses raw features → classifier predicts → returns class label
5. **Screenshots to reference:** `screenshots/vertex_ai/` folder

### 4.5 — Per-Member Deep-Dive Questions to Prepare For

| Member | Their Section | Cross-Questions They Should Prepare For |
|--------|---------------|------------------------------------------|
| Dilusha | EDA & Preprocessing | "Why StandardScaler and not MinMaxScaler?", "What happens if an unseen category appears?" |
| Madusha | Regression | "Why did SGD diverge at lr=0.1?", "What would reduce underfitting?", "Explain bias-variance" |
| Harsha | Customer Segments | "What is data leakage?", "Why Gradient Boosting over Random Forest?", "Explain softmax" |
| Thanura | Delivery Status | "Why is 70% accuracy misleading?", "What is SMOTE?", "Explain the threshold sweep" |
| Chamod | Performance Analysis | "Explain precision vs recall", "Why 5x penalty?", "What is a False Negative?" |
| Dasun | Clustering | "What is the Elbow Method?", "How does K-Means work step-by-step?", "What is Silhouette Score?" |

---

## 5. SLIDE DESIGN GUIDELINES

### Do's ✅

- Use **dark background with light text** or a clean white professional theme — be consistent throughout
- Include **one large visual per slide** (chart, diagram, or figure)
- Use **bullet points of 5 words or fewer** for supporting text
- Ensure all text is **24pt minimum** for readability
- Use **consistent colour coding** across all slides (e.g., green = good, red = bad, blue = info)
- Number all slides
- Use **transitions** between presenters (short pause, introduction of next person)

### Don'ts ❌

- ❌ **No code, no terminal outputs, no raw cell outputs**
- ❌ No walls of text — if you need a paragraph, you're doing it wrong
- ❌ No excessive animations or flying transitions
- ❌ No low-resolution images or squished charts
- ❌ No reading directly from slides — speak to the audience, not the screen

### Suggested Tools

- **Google Slides** or **PowerPoint** — standard, proven tools
- **Canva** — good for modern, professional templates
- **Figures:** Use existing PNGs from `reports/figures/` directly — no need to recreate

---

## 6. REHEARSAL CHECKLIST

### Before Rehearsal

- [ ] All 21 slides created and reviewed
- [ ] All figures exported at high resolution
- [ ] Each presenter has read their speech at least 3 times
- [ ] Timer/stopwatch prepared

### During Rehearsal

- [ ] **Full run-through with timer** — must complete in under 30 minutes
- [ ] Each presenter practises smooth **handoff to the next person**
- [ ] Check that all visuals are **legible from the back of the room**
- [ ] No presenter goes over their allocated time
- [ ] All team members listen to each other's sections (viva preparation)

### Final Check

- [ ] Total time: ≤ 30 minutes ✅
- [ ] All 6 members present ✅
- [ ] No code on any slide ✅
- [ ] Visuals drive every slide ✅
- [ ] All accuracy numbers match the report (81.80%, 70.45%, $98.94, k=4) ✅
- [ ] Viva prep: every member can answer questions about MLflow, packaging, and Vertex AI ✅

---

## 7. AVAILABLE FIGURES & SCREENSHOTS

### Report Figures (`reports/figures/`)

| File | Use In Slide |
|------|-------------|
| `continuous_distributions.png` | Slide 4 — EDA findings |
| `correlation_heatmap.png` | Slide 4 — EDA findings |
| `categorical_distributions.png` | Slide 4 — EDA findings |
| `correlation_engineered.png` | Slide 10 — Feature engineering (optional) |
| `cm_customer_segment.png` | Slide 12 — Customer segment results |
| `cm_delivery_status.png` | Slide 14 — Delivery status results |
| `elbow_silhouette.png` | Slide 18 — Optimal k selection |
| `cluster_pca.png` | Slide 19 — PCA cluster visualization |
| `cluster_boxplots.png` | Slide 19 — Cluster feature profiles |
| `cluster_distributions.png` | Slide 20 — Business insights (optional) |

### MLflow Screenshots (`screenshots/mlflow/`)

| File | Use In |
|------|--------|
| `Main Experiments List View.png` | Viva — "Show me your MLflow" |
| `Regression Experiment Runs.png` | Viva — Regression experiment tracking |
| `Classification Champion Run.png` | Viva — Champion selection evidence |
| `Clustering Run Details.png` | Viva — Clustering experiment |
| `Experiment Comparison - Why Champion Selected 1/2/3.png` | Viva — Model comparison rationale |
| `Logged Model Artifacts.png` | Viva — Serialized model artifacts |

### Vertex AI Screenshots (`screenshots/vertex_ai/`)

| File | Use In |
|------|--------|
| `Vertex AI Endpoints - AuraCart Customer Segment Endpoint Active.png` | Viva — Live endpoint evidence |
| `Endpoint Configuration - Model Deployment Details & Traffic Routing.png` | Viva — Deployment architecture |
| `Endpoint Live Prediction Output - Customer Segment Model.png` | Viva — Successful prediction evidence |

---

## IMPORTANT CORRECTIONS VS GEMINI SAMPLE

> The Gemini-generated sample contained several errors that have been corrected in this workflow:

| Issue | Gemini Said | Corrected Value |
|-------|-------------|-----------------|
| Customer Segment Accuracy | 100% (Cost-Sensitive RF) | **81.80%** (Gradient Boosting) |
| Customer Segment Model | Cost-Sensitive RandomForest | **GradientBoostingClassifier** (n=300, depth=5, lr=0.1) |
| Delivery Status Accuracy | 70.45% or 19.2% (inconsistent) | **70.45%** (Logistic Regression baseline) |
| Test MAE (Regression) | $97.52 | **$98.94** (test), $97.52 is the CV mean |
| Tasks 4.x in slides | Included as Slides 12-13 | **NOT IN SLIDES** — viva only per guidelines |
| Total presentation time | Exceeded 30 min (35 min) | **Exactly 30 min** — 6 presenters balanced |
| Slide count | 13 slides | **21 slides** — more visual, minimal text per slide |
| Presenter assignments for 4.x | Presenter 6 covered Task 4 | **No presenter covers Task 4** — viva only |
| Customer Segment Log-Loss | 0.0087 | **0.5742** (from final report) |
| Customer Segment F1 | 1.0000 | **0.8165** (weighted, from final report) |

---

*Document prepared: April 7, 2026*  
*All data verified against: README.md, ITS2140_ML_Project_Report.md, FINAL_PROJECT_SUMMARY.md, and Presentation Guidelines PDF*
