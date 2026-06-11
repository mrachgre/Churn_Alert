"""
Fix syntax error: move assign_cluster_to_customer() to before the page routing.
The function was incorrectly inserted inside an elif block.
"""
path = "app/streamlit_app.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# The function as inserted (to remove from wrong location)
FUNC_BLOCK = """\ndef assign_cluster_to_customer(input_raw_df: pd.DataFrame) -> int:
    \"\"\"
    Assigns a K-Means cluster to a new customer.
    Re-fits MinMaxScaler on X_test_raw.csv (same columns as training) then predicts.
    Returns cluster ID (int) or -1 if K-Means model unavailable.
    \"\"\"
    km = load_kmeans()
    if km is None:
        return -1
    try:
        X_ref = load_xtest_raw_for_scaler()
        num_cols = X_ref.select_dtypes(include=[np.number]).columns.tolist()
        scaler_km = MinMaxScaler()
        scaler_km.fit(X_ref[num_cols].fillna(X_ref[num_cols].median()))
        input_num = input_raw_df.copy()
        for col in num_cols:
            if col not in input_num.columns:
                input_num[col] = 0
        input_num = input_num[num_cols].fillna(0)
        X_scaled = scaler_km.transform(input_num)
        return int(km.predict(X_scaled)[0])
    except Exception:
        return -1

# Page 7: Customer Lookup"""

if FUNC_BLOCK in content:
    # Step 1: Remove from wrong location, keep the page marker
    content = content.replace(FUNC_BLOCK, "\n# Page 7: Customer Lookup", 1)
    print("[fix] Removed function from wrong location")
else:
    print("[fix] ERROR: function block not found at old location")
    # Show context
    idx = content.find("def assign_cluster_to_customer")
    if idx != -1:
        print("  Found at index", idx)
        print("  Context:", repr(content[idx-100:idx+50]))

# Step 2: Insert before the page routing (before "if page == ...")
PAGE_ROUTING_START = 'if page == "\U0001f4ca Overview":'

FUNC_WITH_NEWLINES = """

def assign_cluster_to_customer(input_raw_df: pd.DataFrame) -> int:
    \"\"\"
    Assigns a K-Means cluster to a new customer.
    Re-fits MinMaxScaler on X_test_raw.csv (same columns as training) then predicts.
    Returns cluster ID (int) or -1 if K-Means model unavailable.
    \"\"\"
    km = load_kmeans()
    if km is None:
        return -1
    try:
        X_ref = load_xtest_raw_for_scaler()
        num_cols = X_ref.select_dtypes(include=[np.number]).columns.tolist()
        scaler_km = MinMaxScaler()
        scaler_km.fit(X_ref[num_cols].fillna(X_ref[num_cols].median()))
        input_num = input_raw_df.copy()
        for col in num_cols:
            if col not in input_num.columns:
                input_num[col] = 0
        input_num = input_num[num_cols].fillna(0)
        X_scaled = scaler_km.transform(input_num)
        return int(km.predict(X_scaled)[0])
    except Exception:
        return -1


"""

if PAGE_ROUTING_START in content:
    content = content.replace(PAGE_ROUTING_START,
                               FUNC_WITH_NEWLINES + PAGE_ROUTING_START, 1)
    print("[fix] Inserted function before page routing")
else:
    print("[fix] ERROR: page routing start not found")

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

# Verify syntax
import ast, sys
try:
    ast.parse(content)
    print("[OK] Syntax valid")
except SyntaxError as e:
    print(f"[FAIL] Syntax error line {e.lineno}: {e.msg}")
    sys.exit(1)

print(f"Total lines: {content.count(chr(10))}")
