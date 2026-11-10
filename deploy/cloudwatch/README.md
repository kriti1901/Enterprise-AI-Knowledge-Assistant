# CloudWatch setup (reference, not applied automatically)

The application publishes custom metrics when `CLOUDWATCH_ENABLED=true`:

| Metric | Unit | Emitted by |
|---|---|---|
| `QueryLatencyMs` | Milliseconds | every `/chat` request |
| `RetrievalHitCount` | Count | number of citations per answer |
| `DocumentsIndexed` | Count | each successful ingestion |
| `IngestionFailures` | Count | each failed ingestion |

Structured JSON logs go to stdout. On ECS, the `awslogs` log driver ships them to a log group.

Files in this folder:
- `alarms.json`: example CloudWatch alarm definitions (CLI input).
- `log-metric-filter.json`: turns `chat` log events with `cached=false` into a metric (optional).

Apply example, after replacing the placeholders:

```bash
aws cloudwatch put-metric-alarm --cli-input-json file://deploy/cloudwatch/alarms.json
```

Required IAM permission for the API task role: `cloudwatch:PutMetricData` (scope it to the namespace with a condition on `cloudwatch:namespace`).
