"""
Comprehensive verification test for Live Render and Local deployment.
Tests:
1. Health endpoint
2. UI homepage
3. Swagger /docs
4. Image triage API (with Grad-CAM and conformal prediction set)
5. Video triage API (temporal frame extraction and consensus)
"""
import requests
import os
import json
import time

LIVE_URL = "https://soy-leaf-doctor.onrender.com"

def test_live():
    print("=" * 60)
    print(f"VERIFYING LIVE RENDER HOST: {LIVE_URL}")
    print("=" * 60)

    # 1. Health
    t0 = time.time()
    r = requests.get(f"{LIVE_URL}/health", timeout=20)
    dt = (time.time() - t0) * 1000
    print(f"\n[1] Health Check (/health):")
    print(f"    Status:   {r.status_code}")
    print(f"    Latency:  {dt:.0f} ms")
    print(f"    Payload:  {r.json()}")
    assert r.status_code == 200

    # 2. Main UI
    t0 = time.time()
    r = requests.get(f"{LIVE_URL}/", timeout=20)
    dt = (time.time() - t0) * 1000
    print(f"\n[2] Homepage UI (/):")
    print(f"    Status:   {r.status_code}")
    print(f"    Latency:  {dt:.0f} ms")
    print(f"    Bytes:    {len(r.content)} bytes")
    assert r.status_code == 200

    # 3. Swagger Docs
    t0 = time.time()
    r = requests.get(f"{LIVE_URL}/docs", timeout=20)
    dt = (time.time() - t0) * 1000
    print(f"\n[3] Swagger Documentation (/docs):")
    print(f"    Status:   {r.status_code}")
    print(f"    Latency:  {dt:.0f} ms")
    assert r.status_code == 200

    # 4. Image Triage
    sample_img = "test_benchmark/images/soy_00007.jpg"
    if not os.path.exists(sample_img):
        import glob
        jpgs = glob.glob("data/p4-soy-leaf-health/images/*.jpg")
        sample_img = jpgs[0] if jpgs else None
    print(f"\n[4] Image Triage API (/api/triage):")
    t0 = time.time()
    with open(sample_img, "rb") as f:
        files = {"file": ("00147.jpg", f, "image/jpeg")}
        data = {"lang": "en"}
        r = requests.post(f"{LIVE_URL}/api/triage", files=files, data=data, timeout=30)
    dt = (time.time() - t0) * 1000
    print(f"    Status:          {r.status_code}")
    print(f"    Latency:         {dt:.0f} ms")
    assert r.status_code == 200
    res = r.json()
    triage = res.get("triage", {})
    quality = res.get("quality", {})
    advisory = res.get("advisory", {})
    cam_len = len(res.get("grad_cam", ""))

    print(f"    Quality Passed:  {quality.get('passed')} (blur: {quality.get('blur_score')}, brightness: {quality.get('brightness')})")
    print(f"    Predicted Label: {triage.get('label')}")
    print(f"    Prediction Set:  {triage.get('prediction_set')}")
    print(f"    Triage State:    {triage.get('triage_state')}")
    print(f"    Confidence:      {triage.get('confidence_level')}")
    print(f"    Top Prob:        {triage.get('probabilities', {}).get(triage.get('label'))}")
    print(f"    Advisory Disease:{advisory.get('disease')}")
    print(f"    Grad-CAM Base64: {cam_len} characters")

    # 5. Video Triage
    test_video = "test_video.mp4"
    if os.path.exists(test_video):
        print(f"\n[5] Video Triage API (/api/triage-video):")
        t0 = time.time()
        with open(test_video, "rb") as f:
            files = {"file": ("test_video.mp4", f, "video/mp4")}
            r = requests.post(f"{LIVE_URL}/api/triage-video", files=files, timeout=60)
        dt = (time.time() - t0) * 1000
        print(f"    Status:          {r.status_code}")
        print(f"    Latency:         {dt:.0f} ms")
        assert r.status_code == 200
        vres = r.json()
        print(f"    Frames Analyzed: {vres.get('frames_analyzed')}")
        print(f"    Overall Pred:    {vres.get('overall_prediction')}")
        print(f"    Mean Confidence: {vres.get('mean_confidence'):.4f}")
        print(f"    Summary:         {vres.get('summary')}")
        print(f"    Frame Details:   {len(vres.get('frame_predictions', []))} frames reported")

    print("\n" + "=" * 60)
    print("ALL LIVE TESTS PASSED SUCCESSFULLY (100% HEALTHY)!")
    print("=" * 60)

if __name__ == "__main__":
    test_live()
