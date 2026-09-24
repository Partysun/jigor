### Unknown route -> 404
GET {{BASE_URL}}/
HTTP 404
[Asserts]
jsonpath "$.error" == "not found"

### Missing state -> 400
POST {{BASE_URL}}/v1/systemone
Content-Type: application/json

{
  "questions": {}
}
HTTP 400
[Asserts]
jsonpath "$.error" == "missing \"state\""

### Missing questions -> 400
POST {{BASE_URL}}/v1/systemone
Content-Type: application/json

{
  "state": "x"
}
HTTP 400
[Asserts]
jsonpath "$.error" == "missing \"questions\" object"

### Unknown question type -> 400
POST {{BASE_URL}}/v1/systemone
Content-Type: application/json

{
  "state": "x",
  "questions": {
    "q": { "type": "alien" }
  }
}
HTTP 400
[Asserts]
jsonpath "$.error" == "question \"q\" has unknown type \"alien\""

### Choice criteria with non-string description -> 400
POST {{BASE_URL}}/v1/systemone
Content-Type: application/json

{
  "state": "x",
  "questions": {
    "q": {
      "type": "choice",
      "instructions": "pick one",
      "criteria": { "a": 42 }
    }
  }
}
HTTP 400
[Asserts]
jsonpath "$.error" matches "must be a string or null"