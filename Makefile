zip_source:
	rm -Rf source.zip
	zip -r source.zip code static config data Dockerfile

# build_dev launches a cloud build for the dev environment
# It needs a build argument (SUBDIR) identifying the subdirectory to build in.
build: auth zip_source
	gcloud builds submit source.zip --region=us-central1 --config=cloudbuild.yaml --substitutions COMMIT_SHA=${USER}

.PHONY: auth
auth:
	gcloud config set project pilabs-dev
	@if gcloud auth application-default print-access-token; then echo "Already logged in"; else gcloud auth application-default login; fi
