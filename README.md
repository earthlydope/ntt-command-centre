# NTT Deal Intelligence

The live application is **`ntt-command-centre/`** — React + TypeScript + D3 front end over a
FastAPI + pandas semantic layer. Its README is the product's documentation:
[`ntt-command-centre/README.md`](ntt-command-centre/README.md).

```bash
cd ntt-command-centre
./run.sh            # API on :8808, app on http://localhost:5178
./run.sh verify     # the regression harness; exit code is the failure count
```

## Deploy

| Half | Where | How |
|---|---|---|
| Front end | Vercel, project `prototype-deal-intelligence`, root `ntt-command-centre/web` | `vercel --prod` from that directory, with `VITE_API_BASE` set to the API URL |
| API | Cloud Run `ntt-deal-intelligence-api` (us-central1) | `gcloud run deploy --source ntt-command-centre` with the `NTT_*` environment variables |

Secrets never live in the repository. Locally the API reads `ntt-command-centre/.env`
(gitignored); on Cloud Run the same names are environment variables. The platform is behind a
single shared password (`NTT_ACCESS_PASSWORD`).
