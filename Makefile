# stop on error, no built in rules, run silently
MAKEFLAGS="-S -s -r"

# get tag and commit information
IMAGE_COMMIT := $(shell git log -1 | head -n 1 | cut -d" " -f2)
IMAGE_TAG := $(shell git tag --contains ${IMAGE_COMMIT})

# set the version from the tag and commit details
IMAGE_VERSION := $(or $(IMAGE_TAG),$(IMAGE_COMMIT))
ifneq ($(shell git status --porcelain),)
    IMAGE_VERSION := $(IMAGE_VERSION)-dirty
endif

# get image id based on tag or commit
IMAGE_VERSION := $(or $(IMAGE_TAG),$(IMAGE_COMMIT))
IMAGE_NAME := "ghcr.io/uwcip-research/shared-engagement-projection"
IMAGE_ID := "${IMAGE_NAME}:${IMAGE_VERSION}"

all: build

.PHONY: build
build:
	@echo "building image for ${IMAGE_ID}"
	docker build -t $(IMAGE_NAME):latest .

.PHONY: buildx
buildx:
	@echo "building multiarch image for ${IMAGE_ID}"
	docker buildx build --platform linux/amd64,linux/arm64 -t $(IMAGE_ID) -t $(IMAGE_NAME):latest .

.PHONY: push
push:
	@echo "pushing $(IMAGE_ID)"
	docker buildx build --push --platform linux/amd64,linux/arm64 -t $(IMAGE_ID) -t $(IMAGE_NAME):latest .

.PHONY: clean
clean:
	@echo "removing built image ${IMAGE_ID}"
	docker image rm -f $(IMAGE_NAME):latest $(IMAGE_ID)
