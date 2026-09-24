### Health / startup check
GET {{BASE_URL}}/healthz
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.model" == "von-1.0.0"