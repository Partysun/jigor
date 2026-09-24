### Rate: score question with dict and plain criteria
POST {{BASE_URL}}/v1/systemone
Content-Type: application/json

{
  "state": "Memory utilization reached 98% with frequent OOM killer invocations.",
  "questions": {
    "severity": {
      "type": "score",
      "instructions": "Assess system degradation level.",
      "criteria": [
        { "what": "Nominal operation; within acceptable variance", "examples": ["no incidents", "p99 < 50ms"] },
        "Elevated resource consumption; degraded performance",
        { "what": "Critical threshold; immediate risk of service termination", "examples": ["OOM killer", "latency > 5s"] }
      ]
    }
  }
}
HTTP 200
[Asserts]
jsonpath "$.answers.severity.score" >= 1
jsonpath "$.answers.severity.score" < 3
jsonpath "$.answers.severity.legend" isObject
jsonpath "$.answers.severity.probabilities" isObject
body contains "Nominal operation; within acceptable variance Examples: no incidents, p99 < 50ms"
body contains "Elevated resource consumption; degraded performance"
body contains "Critical threshold; immediate risk of service termination Examples: OOM killer, latency > 5s"