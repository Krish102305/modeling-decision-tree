"""Modeling - decision tree: clinic no-shows (overbook or hold).
Run:  python decision_tree.py   (expects ../data/clinic_appointments.csv)
Every number on the slides comes from this script.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = HERE
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(os.path.join(HERE, "clinic_appointments.csv"))
df["y"] = (df["no_show"] == "no-show").astype(int)  # 1 = no-show (the "positive" class)
FEATURES = ["lead_time_days", "prior_no_shows", "age", "distance_miles",
            "sms_reminder", "new_patient", "appointment_hour"]
print(f"rows={len(df)}, cols={df.shape[1]-1}, no-shows={df.y.sum()} ({df.y.mean():.1%})")

# 1. 70/30 split, stratified so the no-show rate is the same in both parts
train, test = train_test_split(df, test_size=0.30, stratify=df["y"], random_state=SEED)
print(f"train={len(train)} (no-show {train.y.mean():.1%}), test={len(test)} (no-show {test.y.mean():.1%})")
Xtr, ytr, Xte, yte = train[FEATURES], train["y"], test[FEATURES], test["y"]

# 2. v1: choose depth by cross-validation on the TRAINING set only (test set untouched)
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
print("\nCV accuracy on training set by depth (min_samples_leaf=10):")
cv = {}
for d in range(1, 7):
    m = DecisionTreeClassifier(max_depth=d, min_samples_leaf=10, random_state=SEED)
    cv[d] = cross_val_score(m, Xtr, ytr, cv=skf).mean()
    print(f"  depth {d}: {cv[d]:.3f}")
DEPTH = 3  # sensible depth: readable for a manager; CV curve is flat beyond this
print(f"chosen depth = {DEPTH}")
v1 = DecisionTreeClassifier(max_depth=DEPTH, min_samples_leaf=10, random_state=SEED).fit(Xtr, ytr)
print("\nTree (root first):")
print(export_text(v1, feature_names=FEATURES, show_weights=True))
imp = pd.Series(v1.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("Feature importances:\n", imp[imp > 0].round(3).to_string())

# 3. Evaluate on the test set: baseline (predict "show" for everyone) vs v1
base_pred = np.zeros(len(yte), dtype=int)
v1_pred = v1.predict(Xte)

def report(name, y, p):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    acc = accuracy_score(y, p)
    prec = precision_score(y, p, zero_division=0)
    rec = recall_score(y, p, zero_division=0)
    print(f"{name:9s} TN={tn} FP={fp} FN={fn} TP={tp} | acc={acc:.3f} precision={prec:.3f} recall={rec:.3f}")
    return tn, fp, fn, tp

print("\nTEST SET")
b = report("baseline", yte, base_pred)
v = report("v1", yte, v1_pred)
print(f"v1 training accuracy = {accuracy_score(ytr, v1.predict(Xtr)):.3f} vs test accuracy = {accuracy_score(yte, v1_pred):.3f}")

# 4. Decision rule
def decide(appt, model=v1):
    """appt: dict with the 7 feature names. Returns (leaf_id, 'overbook' | 'hold')."""
    x = pd.DataFrame([appt])[FEATURES]
    leaf = int(model.apply(x)[0])
    return leaf, ("overbook" if model.predict(x)[0] == 1 else "hold")

D = dict(lead_time_days=21, prior_no_shows=1, age=27, distance_miles=9.5, sms_reminder=0, new_patient=1, appointment_hour=15)
E = dict(lead_time_days=3, prior_no_shows=0, age=61, distance_miles=2.0, sms_reminder=1, new_patient=0, appointment_hour=9)
print("\nDecisions for upcoming appointments")
for name, a in [("D", D), ("E", E)]:
    leaf, dec = decide(a)
    p = v1.predict_proba(pd.DataFrame([a])[FEATURES])[0, 1]
    path = v1.decision_path(pd.DataFrame([a])[FEATURES])
    nodes = path.indices
    steps = []
    for n in nodes[:-1]:
        f, t = FEATURES[v1.tree_.feature[n]], v1.tree_.threshold[n]
        val = a[f]
        steps.append(f"{f}={val} {'<=' if val <= t else '>'} {t:.2f}")
    print(f"  {name}: leaf {leaf} (leaf no-show rate {p:.0%}) -> {dec.upper()}   path: " + " ; ".join(steps))

# 5. Value (cost table), on the test appointments
VAL = {"hold_show": 0, "overbook_show": -60, "hold_noshow": -120, "overbook_noshow": 0}
def value(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    # tn: hold & showed (0) ; fp: overbook & showed (-60) ; fn: hold & no-show (-120) ; tp: overbook & no-show (0)
    return tn*VAL["hold_show"] + fp*VAL["overbook_show"] + fn*VAL["hold_noshow"] + tp*VAL["overbook_noshow"]
vb, vv = value(yte, base_pred), value(yte, v1_pred)
n = len(yte)
print(f"\nVALUE on {n} test appointments")
print(f"  baseline: total ${vb:,.0f}  per appointment ${vb/n:,.2f}")
print(f"  v1:       total ${vv:,.0f}  per appointment ${vv/n:,.2f}")
print(f"  improvement per appointment: ${(vv-vb)/n:,.2f}")
VOLUME = 20000  # ASSUMPTION: clinic books 20,000 appointments a year (about 80 a day, 250 days)
print(f"  at an assumed {VOLUME:,} appointments/year: baseline ${vb/n*VOLUME:,.0f}, v1 ${vv/n*VOLUME:,.0f}, gain ${(vv-vb)/n*VOLUME:,.0f}/yr")

# Figure: the tree
fig, ax = plt.subplots(figsize=(14, 6))
plot_tree(v1, feature_names=FEATURES, class_names=["checked-in", "no-show"], filled=True, rounded=True, impurity=False, ax=ax, fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "tree.png"), dpi=200)
print("\nsaved out/tree.png")
