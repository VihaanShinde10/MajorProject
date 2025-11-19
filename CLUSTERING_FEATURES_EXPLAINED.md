# Clustering Features Explained

## 🎯 Overview

The behavioral clustering (Layer 5) groups transactions based on **19 numerical features** extracted by Layer 4. These features capture spending patterns, timing, and recurrence that help identify similar transactions.

---

## 📊 All 19 Features Used for Clustering

### **Category 1: Amount Features (3 features)**

#### **1. `amount_log`**
- **Description:** Log-transformed amount: `log(1 + amount)`
- **Purpose:** Normalizes amount scale (₹10 and ₹100 difference is similar to ₹1000 and ₹10000)
- **Example:**
  - ₹30 → log(31) = 3.43
  - ₹199 → log(200) = 5.30
  - ₹5000 → log(5001) = 8.52
- **Why it matters:** Helps cluster small, medium, and large transactions separately

#### **2. `amount_percentile`**
- **Description:** Where this amount ranks in your spending history (0 to 1)
- **Calculation:** `(number of transactions < this amount) / total transactions`
- **Example:**
  - ₹30 is in 10th percentile → 0.10 (very small for you)
  - ₹500 is in 50th percentile → 0.50 (median spending)
  - ₹5000 is in 95th percentile → 0.95 (unusually large)
- **Why it matters:** Identifies unusual spending (very small or very large for you)

#### **3. `amount_zscore`**
- **Description:** How many standard deviations from your average spending
- **Calculation:** `(amount - mean) / std_deviation`
- **Example:**
  - Your avg spending: ₹500, std: ₹300
  - ₹500 → z-score = 0.0 (exactly average)
  - ₹800 → z-score = 1.0 (1 std above average)
  - ₹200 → z-score = -1.0 (1 std below average)
- **Why it matters:** Detects outliers (unusual transactions for your pattern)

---

### **Category 2: Temporal Features (6 features)**

#### **4. `hour`**
- **Description:** Hour of the day (0-23)
- **Example:**
  - 02-04-2024 **19:17** → hour = 19 (7 PM)
  - 04-04-2024 **15:52** → hour = 15 (3 PM)
- **Why it matters:** Groups transactions by time of day (morning coffee vs evening dinner)

#### **5. `day_of_week`**
- **Description:** Day of the week (0=Monday, 6=Sunday)
- **Example:**
  - Monday → 0
  - Friday → 4
  - Sunday → 6
- **Why it matters:** Identifies weekday vs weekend patterns

#### **6. `day_of_month`**
- **Description:** Day number in the month (1-31)
- **Example:**
  - 1st → 1 (salary day, bill payment day)
  - 15th → 15 (mid-month)
  - 30th → 30 (end of month)
- **Why it matters:** Captures monthly billing cycles, salary patterns

#### **7. `is_weekday`**
- **Description:** Binary flag (1 = weekday, 0 = weekend)
- **Example:**
  - Monday-Friday → 1
  - Saturday-Sunday → 0
- **Why it matters:** Separates work-related spending from leisure

#### **8. `is_commute_window`**
- **Description:** Binary flag (1 = 7-10 AM or 5-8 PM, 0 = other)
- **Example:**
  - 7:30 AM → 1 (morning commute)
  - 6:00 PM → 1 (evening commute)
  - 2:00 PM → 0 (not commute time)
- **Why it matters:** Identifies transport/fuel transactions during commute hours

#### **9. `is_month_start`**
- **Description:** Binary flag (1 = first 5 days of month, 0 = other)
- **Example:**
  - 1st, 2nd, 3rd, 4th, 5th → 1
  - 6th onwards → 0
- **Why it matters:** Captures salary receipts, monthly subscriptions, rent payments

---

### **Category 3: Recurrence Features (4 features)**

#### **10. `is_periodic`**
- **Description:** Binary flag (1 = monthly recurring, 0 = irregular)
- **Detection:** Same merchant + similar amount + ~30 days gap + at least 3 occurrences
- **Example:**
  - NETFLIX ₹199 every 10th → 1 (periodic subscription)
  - Random shopping → 0 (not periodic)
- **Why it matters:** Identifies subscriptions and recurring bills

#### **11. `recurrence_confidence`**
- **Description:** How regular the recurrence is (0 to 1)
- **Calculation:** `1 - (std_deviation_of_gaps / mean_gap)`
- **Example:**
  - Gaps: [30, 30, 31, 29] days → std=0.82, mean=30 → confidence = 0.97 (very regular)
  - Gaps: [25, 35, 28, 32] days → std=4.2, mean=30 → confidence = 0.86 (somewhat regular)
  - Random gaps → confidence = 0.0 (not recurring)
- **Why it matters:** Distinguishes reliable subscriptions from irregular patterns

#### **12. `freq_count_30d`**
- **Description:** Number of similar transactions in last 30 days
- **Example:**
  - Daily coffee → 25-30 (frequent)
  - Weekly grocery → 4-5 (moderate)
  - Monthly bill → 1 (rare)
- **Why it matters:** Identifies daily vs weekly vs monthly patterns

#### **13. `days_since_last`**
- **Description:** Days since last similar transaction
- **Example:**
  - Last coffee was yesterday → 1
  - Last Netflix was 30 days ago → 30
  - First time seeing this → 999 (default)
- **Why it matters:** Helps predict when next occurrence will be

---

### **Category 4: Merchant Features (2 features)**

#### **14. `merchant_frequency`**
- **Description:** Total count of transactions with this merchant in history
- **Example:**
  - Local bakery (visit often) → 50
  - Netflix (monthly) → 6
  - One-time shop → 0
- **Why it matters:** Separates regular merchants from one-time purchases

#### **15. `is_new_merchant`**
- **Description:** Binary flag (1 = first time, 0 = seen before)
- **Example:**
  - First transaction with JULFIKAR → 1
  - 10th transaction with JULFIKAR → 0
- **Why it matters:** New merchants are harder to predict, need special handling

---

### **Category 5: Rolling Statistics (4 features)**

#### **16. `avg_amount_7d`**
- **Description:** Average spending in last 7 days
- **Example:**
  - Last 7 days: ₹500 total / 7 = ₹71.43 per day
- **Why it matters:** Identifies spending bursts vs normal periods

#### **17. `std_amount_7d`**
- **Description:** Standard deviation of amounts in last 7 days
- **Example:**
  - Consistent spending: std = ₹50 (similar amounts)
  - Erratic spending: std = ₹500 (wildly varying)
- **Why it matters:** Detects spending volatility (stable vs chaotic)

#### **18. `txn_count_7d`**
- **Description:** Number of transactions in last 7 days
- **Example:**
  - Busy week: 20 transactions
  - Quiet week: 3 transactions
- **Why it matters:** Captures activity level patterns

#### **19. (Implicit) `merchant_similarity`**
- **Description:** During clustering, merchant frequency acts as similarity metric
- **Note:** Not a separate feature, but merchant patterns are captured in feature #14
- **Why it matters:** Groups transactions from same merchant together

---

## 🎨 How Features Create Clusters

### **Example Cluster 1: Subscriptions**
```
Feature Profile:
  ✅ amount_log: 5.3 (₹199)
  ✅ is_periodic: 1
  ✅ recurrence_confidence: 0.95
  ✅ freq_count_30d: 1
  ✅ days_since_last: 30
  ✅ is_month_start: 1
  ✅ merchant_frequency: 6

Transactions in this cluster:
  - NETFLIX ₹199 (10th of each month)
  - Spotify ₹119 (15th of each month)
  - Prime ₹999 (1st of each month)
```

### **Example Cluster 2: Daily Coffee**
```
Feature Profile:
  ✅ amount_log: 3.4 (₹30)
  ✅ is_periodic: 0 (not monthly, daily!)
  ✅ freq_count_30d: 25
  ✅ days_since_last: 1
  ✅ is_weekday: 1
  ✅ hour: 8-9 (morning)
  ✅ merchant_frequency: 100+

Transactions in this cluster:
  - JULFIKAR BAKERY ₹30 (every morning)
  - Cafe Coffee Day ₹50 (every morning)
  - BIKANER ₹40 (breakfast items)
```

### **Example Cluster 3: Person-to-Person Transfers**
```
Feature Profile:
  ✅ amount_zscore: +1.5 (larger than usual)
  ✅ is_periodic: 0
  ✅ freq_count_30d: 1-2
  ✅ is_new_merchant: 1 (different people)
  ✅ merchant_frequency: 1-3 (occasional)

Transactions in this cluster:
  - VINAYAK ₹943
  - ANUSHKA ₹943
  - SHUBHAM ₹943
  (Similar amounts, different recipients)
```

### **Example Cluster 4: Weekend Entertainment**
```
Feature Profile:
  ✅ amount_log: 6.0 (₹400)
  ✅ is_weekday: 0
  ✅ day_of_week: 6 (Saturday)
  ✅ hour: 14-18 (afternoon)
  ✅ merchant_frequency: 1-2

Transactions in this cluster:
  - IMAGICAA ₹400
  - PVR Cinemas ₹350
  - BookMyShow ₹450
```

---

## 🔧 Feature Engineering Details

### **Why These 19 Features?**

1. **Amount features (3)** → Capture transaction size patterns
2. **Temporal features (6)** → Capture when transactions happen
3. **Recurrence features (4)** → Capture if transactions repeat
4. **Merchant features (2)** → Capture where you spend
5. **Rolling stats (4)** → Capture recent spending context

Total: **19 features** → Enough to capture complex patterns, not too many to cause overfitting

### **Feature Scaling**

Before clustering, all features are standardized:
```python
from sklearn.preprocessing import StandardScaler
scaler.fit_transform(features)  # Mean=0, StdDev=1
```

**Why:** HDBSCAN (clustering algorithm) is sensitive to feature scales. Standardization ensures all features contribute equally.

---

## 📊 How HDBSCAN Uses These Features

### **Step 1: Calculate Distances**
For every pair of transactions, calculate Euclidean distance:
```
distance = sqrt(sum((feature_A[i] - feature_B[i])^2 for all 19 features))
```

### **Step 2: Build Density Graph**
- Find dense regions (many nearby points)
- Sparse regions are marked as "noise"

### **Step 3: Form Clusters**
- Dense regions become clusters
- Each cluster = group of similar transactions
- Noise points (-1 label) = don't fit any cluster

### **Step 4: Label Clusters**
Using semantic categories from earlier layers:
- Cluster 0: Majority are "Food & Dining" → Label: "Food & Dining"
- Cluster 1: Majority are "Subscriptions" → Label: "Subscriptions"
- Cluster 2: Majority are "Transfers" → Label: "Transfers"

---

## 🎯 Real Example from Your Data

### **Transaction: NETFLIX ₹199 on 10th**

**Extracted Features:**
```python
{
    'amount_log': 5.30,           # log(200)
    'amount_percentile': 0.45,    # Below median spending
    'amount_zscore': -0.2,        # Slightly below average
    
    'hour': 19,                   # 7 PM
    'day_of_week': 3,             # Thursday
    'day_of_month': 10,           # 10th
    'is_weekday': 1,              # Weekday
    'is_commute_window': 1,       # 7 PM is commute time
    'is_month_start': 0,          # Not month start
    
    'is_periodic': 1,             # Recurring monthly!
    'recurrence_confidence': 0.98,# Very regular
    'freq_count_30d': 1,          # Once per month
    'days_since_last': 30,        # Last payment 30 days ago
    
    'merchant_frequency': 12,     # 12 months of Netflix
    'is_new_merchant': 0,         # Seen many times
    
    'avg_amount_7d': 250,         # Recent spending
    'std_amount_7d': 150,         # Moderate variance
    'txn_count_7d': 5             # 5 transactions this week
}
```

**Cluster Assignment:**
- HDBSCAN finds this is closest to "Subscription Cluster"
- Other transactions in cluster: Spotify, Prime, YouTube Premium
- **Result:** Classified as "Subscriptions" ✅

---

## 🔍 Viewing Features in Action

### **In the Clusters Tab:**

1. **Cluster Overview** shows:
   - How many clusters formed
   - Size of each cluster
   - Noise ratio

2. **Cluster Details** shows:
   - Cohesion score (how tight the cluster is)
   - Category label (from majority voting)

3. **Quality Metrics** show:
   - **Silhouette Score** (are clusters well-separated?)
   - **Davies-Bouldin Index** (how distinct are clusters?)

4. **2D Visualization** shows:
   - PCA projection of all 19 features → 2D
   - Visual representation of clusters

---

## 💡 Tips for Better Clustering

### **1. Need History for Good Features**
- Minimum 20 transactions
- Ideal: 50-100+ transactions
- More data = better recurrence detection

### **2. Consistent Merchant Names**
- "JULFIKAR" every time (good)
- "JULFIKAR", "JULFIKAR BAKERY", "JULFI" (confusing)
- Use recipient_name for consistency

### **3. Date/Time Accuracy Matters**
- Correct timestamps enable temporal features
- Wrong dates → poor clustering

### **4. Amount Precision**
- Exact amounts help recurrence detection
- ₹199.00 every month → perfect pattern
- ₹195-205 varying → weaker pattern

---

## 📈 Expected Cluster Distribution

For typical spending (100+ transactions):

```
Cluster -1 (Noise): 10-20%
  - One-time purchases
  - Unusual transactions
  - New merchants

Cluster 0 (Subscriptions): 5-10%
  - Netflix, Spotify, etc.
  - Monthly recurring bills

Cluster 1 (Daily Essentials): 30-40%
  - Coffee, bakery, groceries
  - High frequency, small amounts

Cluster 2 (Transfers): 15-25%
  - Person-to-person
  - Similar amounts, different recipients

Cluster 3+ (Other Patterns): 20-30%
  - Transport, entertainment, etc.
  - Various patterns
```

---

## 🎉 Summary

**The 19 features capture:**
- ✅ **How much** you spend (amount features)
- ✅ **When** you spend (temporal features)
- ✅ **How often** you spend (recurrence features)
- ✅ **Where** you spend (merchant features)
- ✅ **Recent trends** (rolling statistics)

**Together, they enable HDBSCAN to:**
- Group similar transactions
- Identify patterns (subscriptions, daily habits, transfers)
- Learn your spending behavior
- Improve predictions over time

**All automatically! 🚀**

