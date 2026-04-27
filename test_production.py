import requests
import time
import sys

def test_system():
    base_url = "http://127.0.0.1:8000"
    
    print("🚀 Starting Production System Audit...\n")

    # 1. Health Check
    try:
        r = requests.get(f"{base_url}/api/health", timeout=5)
        if r.status_code == 200:
            print("✅ Health Check: OK")
        else:
            print(f"❌ Health Check: Failed (Status {r.status_code})")
    except:
        print("⚠️ Server not running at 127.0.0.1:8000. Start it with 'python -m app.main'")
        return

    # 2. Test Hinglish Retrieval
    print("\n🔍 Testing Hinglish Retrieval...")
    payload = {"message": "BCA ki fees kya hai?"}
    t1 = time.time()
    try:
        r = requests.post(f"{base_url}/api/chat", json=payload, timeout=15)
        if r.status_code == 200:
            data = r.json()
            print(f"✅ Hinglish Response Received ({time.time()-t1:.2f}s)")
            print(f"AI Answer: {data['answer'][:100]}...")
        else:
            print(f"❌ Hinglish Test Failed: {r.status_code}")
    except Exception as e:
        print(f"❌ Hinglish Test Error: {e}")

    # 3. Test English Retrieval
    print("\n🔍 Testing English Retrieval...")
    payload = {"message": "Tell me about the admission process for MBA."}
    t1 = time.time()
    try:
        r = requests.post(f"{base_url}/api/chat", json=payload, timeout=15)
        if r.status_code == 200:
            data = r.json()
            print(f"✅ English Response Received ({time.time()-t1:.2f}s)")
            print(f"AI Answer: {data['answer'][:100]}...")
        else:
            print(f"❌ English Test Failed: {r.status_code}")
    except Exception as e:
        print(f"❌ English Test Error: {e}")

    # 4. Test PDF Trigger
    print("\n🔍 Testing PDF Trigger...")
    payload = {"message": "Can I see the placement brochure?"}
    try:
        r = requests.post(f"{base_url}/api/chat", json=payload, timeout=15)
        data = r.json()
        if data.get("pdf_url"):
            print(f"✅ PDF Trigger: SUCCESS ({data['pdf_url']})")
        else:
            print("⚠️ PDF Trigger: No URL returned (might be data missing in Qdrant)")
    except Exception as e:
        print(f"❌ PDF Test Error: {e}")

    print("\n✅ System Audit Complete.")

if __name__ == "__main__":
    test_system()
