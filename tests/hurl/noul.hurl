### Judge: noul question, spec wire example (state as dict -> pretty JSON)
POST {{BASE_URL}}/v1/systemone
Content-Type: application/json

{
  "model": "von-1.0.0",
  "state": { "error": "Disk volume /var/log at 98% capacity." },
  "questions": {
    "requires_intervention": {
      "type": "noul",
      "instructions": "Does this disk space condition require operational intervention?"
    }
  }
}
HTTP 200
[Asserts]
jsonpath "$.model" == "von-1.0.0"
jsonpath "$.answers.requires_intervention.noul" > 0.5
jsonpath "$.answers.requires_intervention.noul" < 1

### Judge with explicit true/false criteria
POST {{BASE_URL}}/v1/systemone
Content-Type: application/json

{
  "state": "Disk volume /var/log at 98% capacity.",
  "questions": {
    "intervention": {
      "type": "noul",
      "instructions": "Does this disk space condition require operational intervention?",
      "criteria": { "true": "Yes, it requires action", "false": "No, it can wait" }
    }
  }
}
HTTP 200
[Asserts]
jsonpath "$.answers.intervention.noul" > 0.5
jsonpath "$.answers.intervention.noul" < 1