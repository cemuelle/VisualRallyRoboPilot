# How to deploy image

1. Push the latest version of the code (the game instances) on the github https://github.com/cemuelle/VisualRallyRoboPilot/ on the right branch

2. build the docker image with the command:
	```
	docker-compose build
	```

3. upload the built image on docker hub


## Notes:

1. the selected branch can be changed in Dockerfile.headless at line 30