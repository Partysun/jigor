### Routing: known model answers with the resolved provider
POST {{BASE_URL}}/v1/systemone
Content-Type: application/json

{
  "model": "von-1.0.0",
  "state": "Disk volume /var/log at 98% capacity.",
  "questions": {
    "requires_intervention": {
      "type": "noul",
      "instructions": "Does this disk space condition require operational intervention?"
    }
  }
}
HTTP 200
[Asserts]
jsonpath "$.backend" == "local"
jsonpath "$.model" == "von-1.0.0"
jsonpath "$.answers.requires_intervention.type" == "noul"
jsonpath "$.answers.requires_intervention.noul" > 0

### Routing: jeff (local, lazy-loaded ONNX)
POST {{BASE_URL}}/v1/systemone
Content-Type: application/json

{
  "model": "jeff",
  "state": {"error": "Disk volume /var/log at 98% capacity."},
  "questions": {
    "requires_intervention": {
      "type": "noul",
      "instructions": "Does this disk space condition require operational intervention?"
    },
    "team": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {"1": "Refunds and payments", "2": "Damaged or lost parcels", "3": "Account and login problems"}
    }
  }
}
HTTP 200
[Asserts]
jsonpath "$.backend" == "local"
jsonpath "$.model" == "jeff-qwen3.5-0.8b"
jsonpath "$.answers.requires_intervention.type" == "noul"
jsonpath "$.answers.requires_intervention.noul" > 0
jsonpath "$.answers.team.type" == "choice"
jsonpath "$.answers.team.choice" == "2"
jsonpath "$.answers.team.probabilities.2" > 0.5

### Unknown model id -> 400
POST {{BASE_URL}}/v1/systemone
Content-Type: application/json

{
  "model": "no-such-model",
  "state": "hello",
  "questions": {
    "q": {"type": "noul", "instructions": "Is this a question?"}
  }
}
HTTP 400
[Asserts]
jsonpath "$.error" exists