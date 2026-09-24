### Fan-out: one request with all three question types
POST {{BASE_URL}}/v1/systemone
Content-Type: application/json

{
  "model": "von-1.0.0",
  "state": { "error": "Database replication lag exceeded 45 seconds." },
  "questions": {
    "domain": {
      "type": "choice",
      "instructions": "Classify the root cause domain of this incident.",
      "criteria": ["infrastructure", "billing", "feature_request"]
    },
    "blocking": {
      "type": "noul",
      "instructions": "Is this incident actively blocking customer operations?"
    },
    "impact": {
      "type": "score",
      "instructions": "Assess the severity of this incident.",
      "criteria": [
        { "what": "Nominal", "examples": ["no incidents"] },
        "Elevated resource consumption; degraded performance",
        { "what": "Critical threshold", "examples": ["OOM killer", "latency > 5s"] }
      ]
    }
  }
}
HTTP 200
[Asserts]
jsonpath "$.model" == "von-1.0.0"
jsonpath "$.answers.domain.choice" == "infrastructure"
jsonpath "$.answers.blocking.noul" >= 0
jsonpath "$.answers.blocking.noul" <= 1
jsonpath "$.answers.impact.score" >= 0
jsonpath "$.answers.impact.legend" isObject
body contains "Critical threshold Examples: OOM killer, latency > 5s"