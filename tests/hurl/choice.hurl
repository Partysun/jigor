### Choice with object criteria (name -> description)
POST {{BASE_URL}}/v1/systemone
Content-Type: application/json

{
  "state": "Database replication lag on cluster us-west-2 exceeded 45 seconds.",
  "questions": {
    "domain": {
      "type": "choice",
      "instructions": "Classify the root cause domain of this incident.",
      "criteria": {
        "infrastructure": "Database, hardware, network, or server failures",
        "billing": "Invoices, payments, refunds, subscription queries",
        "feature_request": "Requests for new platform capabilities"
      }
    }
  }
}
HTTP 200
[Asserts]
jsonpath "$.model" == "von-1.0.0"
jsonpath "$.answers.domain.choice" == "infrastructure"
jsonpath "$.answers.domain.confidence" == 0.428
jsonpath "$.answers.domain.probabilities.infrastructure" == 0.6203
jsonpath "$.answers.domain.probabilities.billing" == 0.1873
jsonpath "$.answers.domain.probabilities.feature_request" == 0.1924

### Choice with list criteria (names only)
POST {{BASE_URL}}/v1/systemone
Content-Type: application/json

{
  "state": "Database replication lag on cluster us-west-2 exceeded 45 seconds.",
  "questions": {
    "domain": {
      "type": "choice",
      "instructions": "Classify the root cause domain of this incident.",
      "criteria": ["infrastructure", "billing", "feature_request"]
    }
  }
}
HTTP 200
[Asserts]
jsonpath "$.answers.domain.choice" == "infrastructure"
jsonpath "$.answers.domain.probabilities.infrastructure" > 0.3
jsonpath "$.answers.domain.probabilities.infrastructure" < 0.6