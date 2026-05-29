"""
P0-A: 传统ML基线实验（XGBoost + Ridge Regression）
对比 T2 核减率预测（MAE）
从 govreview_bench_v2_desensitized.json 提取结构化特征
"""
import json
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.dummy import DummyRegressor
import warnings
warnings.filterwarnings('ignore')

# ─── 加载数据 ────────────────────────────────────────────────
DATA_PATH = "/Volumes/Elements SE/科研/软件学报-软件投资审核/核减Agent/bench/bench_data/govreview_bench_v2_desensitized.json"
with open(DATA_PATH) as f:
    data = json.load(f)

print(f"Total samples: {len(data)}")

# ─── 特征工程 ────────────────────────────────────────────────
def extract_features(item):
    """从结构化字段中提取数值特征，不使用任何文本内容"""
    features = {}
    
    # 1. 费用类别（sheet_name），one-hot via label
    features['sheet_name'] = item.get('sheet_name', '') or ''
    
    # 2. 金额量级特征
    try:
        orig = float(item.get('original_total') or 0)
    except:
        orig = 0.0
    features['original_total'] = orig
    features['log_original'] = np.log1p(orig)
    features['log10_original'] = np.log10(orig + 1)
    
    # 3. 单价（如果有）
    try:
        up = float(item.get('unit_price') or 0)
    except:
        up = 0.0
    features['unit_price'] = up
    features['has_unit_price'] = float(up > 0)
    
    # 4. 数量（如果有）
    try:
        qty = float(item.get('quantity') or 0)
    except:
        qty = 0.0
    features['quantity'] = qty
    features['has_quantity'] = float(qty > 0)
    
    # 5. 有无证据文本
    features['has_evidence'] = float(bool(item.get('has_evidence')) or 
                                      bool(item.get('evidence_chunks')))
    
    # 6. 难度等级
    diff_map = {'L1': 0, 'L2': 1, 'L3': 2, 'L4': 3}
    features['difficulty'] = diff_map.get(item.get('difficulty', 'L2'), 1)
    
    # 7. 项目ID（project_id）
    features['project_id'] = item.get('project_id', 'P01')
    
    # 8. 金额区间特征
    features['amount_bin'] = int(np.digitize(orig, [0, 100, 500, 1000, 5000, 10000]))
    
    return features

# 提取特征和标签
raw_features = [extract_features(item) for item in data]
y = np.array([float(item.get('ground_truth_rate') or 0) for item in data])

# label encode 分类特征
le_sheet = LabelEncoder()
le_proj = LabelEncoder()
sheet_names = [f['sheet_name'] for f in raw_features]
proj_ids = [f['project_id'] for f in raw_features]
le_sheet.fit(sheet_names)
le_proj.fit(proj_ids)

X = np.column_stack([
    le_sheet.transform(sheet_names),             # 费用类别
    le_proj.transform(proj_ids),                  # 项目ID
    [f['original_total'] for f in raw_features],
    [f['log_original'] for f in raw_features],
    [f['log10_original'] for f in raw_features],
    [f['unit_price'] for f in raw_features],
    [f['has_unit_price'] for f in raw_features],
    [f['quantity'] for f in raw_features],
    [f['has_quantity'] for f in raw_features],
    [f['has_evidence'] for f in raw_features],
    [f['difficulty'] for f in raw_features],
    [f['amount_bin'] for f in raw_features],
])

print(f"Feature matrix shape: {X.shape}")
print(f"Target (reduction rate) stats: mean={y.mean():.2f}, std={y.std():.2f}, "
      f"min={y.min():.2f}, max={y.max():.2f}")

# ─── 10折交叉验证 ───────────────────────────────────────────
kf = KFold(n_splits=10, shuffle=True, random_state=42)

def cv_mae(model, X, y, label):
    maes = []
    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        
        model_copy = type(model)(**model.get_params()) if hasattr(model, 'get_params') else model
        model_copy.fit(X_tr_s, y_tr)
        pred = model_copy.predict(X_te_s)
        # clip to [0, 100]
        pred = np.clip(pred, 0, 100)
        maes.append(mean_absolute_error(y_te, pred))
    
    mean_mae = np.mean(maes)
    std_mae = np.std(maes)
    print(f"{label}: MAE = {mean_mae:.2f} ± {std_mae:.2f}")
    return mean_mae, std_mae

print("\n=== 10-fold Cross-Validation MAE (T2: 核减率预测) ===")

# 1. 统计基线（训练集均值盲猜）
dummy = DummyRegressor(strategy='mean')
dummy_maes = []
for tr, te in kf.split(X):
    dummy.fit(X[tr], y[tr])
    dummy_maes.append(mean_absolute_error(y[te], np.clip(dummy.predict(X[te]), 0, 100)))
stat_mae = np.mean(dummy_maes)
print(f"统计基线（均值预测）: MAE = {stat_mae:.2f} ± {np.std(dummy_maes):.2f}")

# 2. Ridge Regression
ridge = Ridge(alpha=1.0)
ridge_mae, ridge_std = cv_mae(ridge, X, y, "Ridge Regression")

# 3. Gradient Boosting (XGBoost-like)
gbm = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, 
                                  min_samples_leaf=3, random_state=42)
gbm_mae, gbm_std = cv_mae(gbm, X, y, "Gradient Boosting (GBM)")

# 4. Random Forest
rf = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=3,
                            random_state=42, n_jobs=-1)
rf_mae, rf_std = cv_mae(rf, X, y, "Random Forest")

# ─── 汇总结果 ───────────────────────────────────────────────
print("\n=== 与LLM结果对比汇总 ===")
print("方法                          T2 MAE   备注")
print("-" * 60)
print(f"统计基线（均值）               {stat_mae:.2f}     论文中已报告=21.89")
print(f"Ridge Regression              {ridge_mae:.2f}     传统线性ML")
print(f"Random Forest                 {rf_mae:.2f}     传统树集成ML")
print(f"Gradient Boosting             {gbm_mae:.2f}     传统提升树ML")
print(f"DeepSeek V3 + BM25（LLM）    21.36     论文主实验最优")
print(f"Qwen-Plus + BM25（LLM）      22.51     论文主实验结果")

# 保存结果
results = {
    "stat_baseline": {"mae": round(stat_mae, 2), "std": round(np.std(dummy_maes), 2)},
    "ridge": {"mae": round(ridge_mae, 2), "std": round(ridge_std, 2)},
    "random_forest": {"mae": round(rf_mae, 2), "std": round(rf_std, 2)},
    "gradient_boosting": {"mae": round(gbm_mae, 2), "std": round(gbm_std, 2)},
    "llm_deepseek_bm25": {"mae": 21.36},
    "llm_qwen_bm25": {"mae": 22.51},
    "llm_glm4_bm25": {"mae": 21.78},
    "llm_doubao_bm25": {"mae": 22.48},
}
import json as j2
out = "/Volumes/Elements SE/科研/软件学报-软件投资审核/核减Agent/bench_tasks/results/ml_baseline_results.json"
with open(out, 'w') as f:
    j2.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n✅ Results saved to: {out}")
