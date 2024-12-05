# To use the current solution:

1. get this directory on the calypso side

2. deploy the endpoint.yaml file with the command: 
	```
	kubectl apply -f endpoint.yaml -n isc3
	```

	(the endpoint can stay between tests, it doesn't need to be touched between tests/modifications as long as the appname doesn't change in the headless.yaml file)

3. use the following command to start game instances on pods, with x being the number of game instances / pods you want running:
	```
	bash kubeStart.sh x
	```

4. once you are done, you can delete the pods with the follonwing command:
	```
	bash kubeEnd.sh
	```

## Notes:

1. The docker image on each pod is the latest image uploaded on https://hub.docker.com/repository/docker/sebdufond/headless-rally/general.

	To see how to change this image, go to ../docker/

2. The pods must be redeployed with steps 4 and 3 if any change is done on the docker image for it to apply
