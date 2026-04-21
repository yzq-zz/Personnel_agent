"""
测试阿里云 DashScope embedding API 连接
"""
import os
import requests

API_KEY = os.environ.get("QWEN_API_KEY", "sk-b999c0353f9642a39d690363dc953239")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "text-embedding-v3"

def test_embedding():
    url = f"{BASE_URL}/embeddings"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "input": "你好",
    }

    print(f"URL: {url}")
    print(f"API_KEY: {API_KEY[:10]}...{API_KEY[-4:]}")
    print(f"Model: {MODEL}")
    print()

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        print(f"Status: {resp.status_code}")
        print(f"Headers: {dict(resp.headers)}")
        print()
        print(f"Response: {resp.text[:2000]}")
    except Exception as e:
        print(f"Request failed: {e}")

def test_with_sync():
    """直接用 httpx/requests 测试同步调用"""
    import httpx
    url = f"{BASE_URL}/embeddings"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "input": ["你好", "今天天气怎么样"],
    }

    print("\n--- httpx sync test ---")
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.post(url, headers=headers, json=payload)
            print(f"Status: {resp.status_code}")
            print(f"Response: {resp.text[:2000]}")
    except Exception as e:
        print(f"httpx failed: {e}")

if __name__ == "__main__":
    test_embedding()
    test_with_sync()