![](https://gitlab.hevs.ch/uploads/-/system/project/avatar/1282/rallyrobopilot.jpg)

# Rally Robopilot Project

This repository holds a sandbox driving simulation controllable via a network interface as a machine learning and data collection challenge.  

# Installation
From the root of the repository, run
```
pip install -r requirements.txt
pip install -e .
```

To run the game, you can use
```
python scripts/main.py
```

# Generality
Launching main.py starts a race with a single car on the provided track. 
This track can be controlled either by keyboard (*AWSD*) or by a socket interface. 
An example of such interface is included in the code in *example_data_collector.py*.

# Sensing
The car sensing is available in two commodities: **raycasts** and **images**. These sensing snapshots are sent at 10 Hertz (i.e. 10 times a second). Due to this fact, correct recetion of snapshot messages has to be done regularly (See Server buffer saturation section).

# Communication protocol

A remote controller can be impemented using TCP socket connecting on localhost on port 7654. 
Different commands can be issued to the race simulation to control the car.

The car simulator expect specific commands to be sent. A command is a specific set of words or numbers. 
**Spaces** are used as elements separator in a single command. **Semi-colon (;)** as command separator.

##  Car controls
```
push|release forward|right|left|back;
```
To simulate key press and control the car.

## Reset controls
```
set position x,y,z;
```
To set the reset position at the (x,y,z) location. x,y,z are float numbers with english style dot separator (e,g. 3.1415, 2.5, 6.34234) themselves separated by a comma.
```
set rotation a;
```
to set the car orientation, with a being a float number in degrees
```
set speed v;
```
to set the starting speed after reset

```
reset;
```
to reset the car at the provided location.

## Ray sensing

```
set ray visible|hidden;
```
To toggle the ray sensor visibility. In direct play **v** can be used to toggle ray visibility

##  /!\ Server buffer saturation
While implementing your own controller, make sure to regularly empty the socket buffer connected to the server by regularly calling **NetworkDataCmdInterface.recv_msgs**. 
Otherwise, while sending images, the server might fill the buffer and throw socket errors leading to a crash. 

# Multiplayer

To run multiplayer, run the `main.py` file and click `Multiplayer`. Then enter the ip address (this can be defaulted to 'localhost') and the port (default: 25565). Click `Create - Server` and then click `Join - Server`.
For others to join, run the `main.py`, click `Multiplayer` and then click `Join Server`. Enter the public ip address and the port of the server. Click `JOIN` and then you're in. Unlimited people can join!

If you don't want to play on a server, run the `main.py` file and click `Singleplayer`.

# Controls

W - Drive
S - Brake
A, D - Turn
SPACE - Hand Brake
ESCAPE - Pause Menu
G - Respawn

# Run in Docker 
We provide two docker containers to run the game: one that launches the usual graphic version (only runs on Linux host with X11), 
and one that runs the game on a headless server (will be used for model training).


The graphical container can be run using


`docker compose run gui`


and the headless one can be run using 


`docker compose run headless`


# Credits
This code is based on the repository [https://github.com/mandaw2014/Rally](https://github.com/mandaw2014/Rally)
