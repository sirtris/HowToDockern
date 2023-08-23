# Getting started
[My Dockerhub](https://hub.docker.com/r/hannahkniesel/docker_gpu/tags).

Clone the repository and setup the correct folder structure.
```bash
git clone https://github.com/HannahKniesel/HowToDockern.git
cd HowToDockern
mkdir data
mkdir output
```

Build the docker image 
```bash
docker build -t <image-name> .
```

Run the docker image (locally - for development)
```bash
docker run --gpus all -it --rm --ipc=host -v /mnt/e/Documents/UbuntuCode/1_DockerTest/HowToDockern/workspace/:/repository/workspace/ --name <container-name> <image-name>
```

Push the docker to dockerhub
```bash
docker tag <image-name> <user-name>/<image-name>:<version>
docker push <user-name>/<image-name>:<version>

```

Use on cluster
> **__NOTE__** Make sure that your startscript.sh lies in the directory ´/code/´ and you submit from this directory. Also, your code should be developed, to be started from this directory.

Your startscript.sh should look like this: 
```bash
python classifier.py

```

Then you can submit it as follows:
```bash
submit ./startscript.sh --name <name> --custom <user-name>/<image-name>:<version>
```
