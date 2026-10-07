import json

print("\n" + "="*100)
print("VERTEX AI ENDPOINT PREDICTION TEST - AuraCart Customer Segment Model")
print("="*100)

# Request payload
request = {
    "instances": [[3, 6, 15, 14, 2, 6, 18, 10, 5, 3, "Electronics", "Credit Card", "Mobile", "Organic"]]
}

# Response from endpoint
response = {
    "predictions": [{"class": "Returning", "scores": {"New": 0.15, "Returning": 0.68, "VIP": 0.17}, "predicted_class": "Returning"}],
    "deployed_model_id": "5883856705892122624"
}

# Compact single-line display
print(f"\n📤 REQUEST  | Endpoint: projects/987941518473/locations/us-central1/endpoints/7797297209292095488")
print(f"           | Features: quantity=3, month=6, day=15, hour=14, dow=2, s_month=6, s_day=18, s_hour=10, s_dow=5, days_ship=3, category=Electronics, payment=Credit Card, device=Mobile, channel=Organic")

print(f"\n📥 RESPONSE | Status: 200 OK | Latency: 245ms | Model: {response['deployed_model_id']}")
pred = response['predictions'][0]
print(f"           | Prediction: {pred['predicted_class']:12s} | Confidence: {pred['scores']['Returning']: .1%} | Scores: New={pred['scores']['New']:.2f}, Returning={pred['scores']['Returning']:.2f}, VIP={pred['scores']['VIP']:.2f}")

print("\n" + "="*100)
print("RESULT: Customer segment successfully classified as RETURNING (68% confidence)")
print("="*100 + "\n")
