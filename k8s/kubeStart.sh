#!/bin/bash
export K8S_NUMBER_PODS=$1
envsubst < headlessIsol.yaml | kubectl apply -f - -n isc3
