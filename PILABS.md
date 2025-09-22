## Pi Labs

This fork as been altered in a few ways:

1. It now runs on GCP Cloud Run, roughly here:
   https://console.cloud.google.com/run/detail/us-central1/nlweb/observability/metrics?project=pilabs-dev
1. The local Dockerfile has been altered to support the above (packaging everything into it by default)
1. If WITHPI_API_KEY is set, which it is in Cloud Run, activate Pi Scorer instead of LLM scoring.
   1. This removes the relevance threshold, since we haven't trained for that.
   1. It uses the result's Schema.org snippets, rather than LLM-generated snippets.
