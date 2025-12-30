import requests

# Test health endpoint
print("== Health ==")
try:
    r = requests.get("http://127.0.0.1:8000/health")
    print(r.status_code, r.json())
except Exception as e:
    print("Health check failed:", e)

# Test analyze endpoint
print("\n== Analyze ==")
try:
    sample = {"text": "we may share personal information with third parties"}
    r = requests.post("http://127.0.0.1:8000/analyze", json=sample)
    print(r.status_code, r.json())
except Exception as e:
    print("Analyze failed:", e)
