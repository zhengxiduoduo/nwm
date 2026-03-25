import pickle
from pprint import pprint

path = "navigation_eval.pkl"

with open(path, "rb") as f:
    data = pickle.load(f)

print("TYPE:", type(data))

# 如果是 list
if isinstance(data, list):
    print("LEN:", len(data))
    if len(data) == 0:
        raise SystemExit("Empty list")

    # 统计前 N 个元素的类型分布
    N = min(50, len(data))
    type_counts = {}
    for i in range(N):
        t = type(data[i]).__name__
        type_counts[t] = type_counts.get(t, 0) + 1
    print("HEAD TYPE COUNTS (first", N, "):", type_counts)

    # 打印前几个元素（用 pprint 展开）
    K = min(5, len(data))
    for i in range(K):
        print(f"\n--- ITEM {i} / type={type(data[i])} ---")
        pprint(data[i], width=120)

    # 如果元素是 dict/list/tuple，进一步展示其“内部结构”
    x = data[0]
    if isinstance(x, dict):
        print("\nFIRST ITEM KEYS:", list(x.keys())[:30])
    elif isinstance(x, (list, tuple)):
        print("\nFIRST ITEM LEN:", len(x))
        print("FIRST ITEM[0] TYPE:", type(x[0]) if len(x) else None)

# 如果是 dict（以防万一）
elif isinstance(data, dict):
    print("KEYS:", list(data.keys())[:50])
    k0 = next(iter(data.keys()), None)
    if k0 is not None:
        print("\n--- VALUE OF FIRST KEY ---")
        pprint(data[k0], width=120)

else:
    # 其他类型直接 pprint
    pprint(data, width=120)