#!/bin/bash

# Quick curl test for NimbusGuard Heavy Processing API
# Make this file executable: chmod +x quick_test.sh

BASE_URL="http://localhost:8000"

echo "🎯 Heavy Processing API Test"
echo "============================"

echo -e "\n1. Health Check:"
curl -s $BASE_URL/health | jq '.'

echo -e "\n2. API Stats:"
curl -s $BASE_URL/stats | jq '.'

echo -e "\n3. Heavy Processing Test (this will take several seconds):"
echo "   Sending heavy processing request..."

curl -s -X POST $BASE_URL/process \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "user_id": 12345,
      "dataset": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
      "text_data": "Heavy processing test data with lots of text to process",
      "numerical_values": [99.1, 88.2, 77.3, 66.4, 55.5],
      "operation_type": "extreme_heavy_compute",
      "metadata": {
        "source": "curl_test",
        "complexity": "very_high",
        "batch_id": 42
      },
      "string_array": ["process", "this", "heavy", "data", "load"],
      "nested_data": {
        "level1": {
          "level2": {
            "values": [1,2,3,4,5,6,7,8,9,10]
          }
        }
      }
    },
    "request_id": "curl_heavy_test_001"
  }' | jq '.'

echo -e "\n4. Final Stats Check:"
curl -s $BASE_URL/stats | jq '.'

echo -e "\n✅ Heavy processing test completed!"
echo "📝 Check app.log for detailed processing logs"
echo "⚡ Each request performs 4000+ computational operations"
