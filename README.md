# How to Docker
Find more information [here](https://hackernoon.com/docker-overview-a-complete-guide-43decd218eca).

- Dockerfile ... File with instructions to build the image

- Docker Image ... Image that can be shared across platforms to run code

- Docker Container ... The "environment" which is used to actually run the code

An example project can be found [here](github.link)

## Best Practices with Docker
For new projects, I recommend following structure: 
```
├── workspace
│   ├── code
│   │   ├── all your code
│   ├── data
│   │   ├── all your data
│   ├── output
│   │   ├── logging + checkpoints
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── README.md
└── .gitignore
```
- Like this, you can easily mount/add your code and data directory seperately. 
- The requirements.txt should have all python packages with version control. It can hence be accessed in the *Dockerfile* to install dependencies. 
- Using git + wandb in combintation with docker will help you to keep track of the docker you used for the experiment, as you can access the git commit from your wandb experiments, and then link these to the corresponding dockerfile.  
- You should keep some additional file to write down the results of your experiments. There you should also link the corresponding wandb projects, groups or runs. If you like you can add version control to your docker images and link them in this file as well. (So if you ever want to redo an experiment, you just go to the wandb run, copy the command to run it, get your docker image, and do so.)

## Generate a docker file
More details [here](https://docs.docker.com/engine/reference/builder/#env).

Generate a file *Dockerfile* (**without extension**) in your project directory. 
This way, when you use git + wandb you can find the corresponding commit of one experiment and the correlated Dockerfile. 

Simple example for docker of pytorch extension: 
```docker
FROM pytorch/pytorch:latest
ADD requirements.txt /workspace/ 
RUN pip install -r /workspace/requirements.txt
ENV CUBLAS_WORKSPACE_CONFIG=:16:8 
RUN apt-get update && apt-get upgrade -y
```

[`FROM`](https://docs.docker.com/engine/reference/builder/#from)
Sets the Base Image for subsequent instructions. 
A valid *Dockerfile* must start with a `FROM` instruction.

[`RUN`](https://docs.docker.com/engine/reference/builder/#run)
Executes any commands on top of the new image and commits the results. The resulting image will be used in the next steps of the *Dockerfile*.

[`ENV`](https://docs.docker.com/engine/reference/builder/#env)
The `ENV` instruction sets the environment variable `<key>` to the value `<value>` (such as `ENV <key>=<value>`)

[`ADD`](https://docs.docker.com/engine/reference/builder/#add)
The `ADD` instruction copies new files, directories or remote file URLs from `<src>` and adds them to the filesystem of the image at the path `<dest>` (such as `ADD <src>,... <dest>`). This is helpful when probably for a final release the code/data or similar should be *included* in the docker image itself. 

[`COPY`](https://docs.docker.com/engine/reference/builder/#copy)
The `COPY` instruction copies new files or directories from `<src>` and adds them to the filesystem of the container at the path `<dest>` (such as `COPY <src>,... <dest>`)

## Develop with docker

### Make your own *Dockerfile*

```docker
FROM pytorch/pytorch:latest
ADD requirements.txt /workspace/ 
RUN pip install -r /workspace/requirements.txt
ENV CUBLAS_WORKSPACE_CONFIG=:16:8 
RUN apt-get update && apt-get upgrade -y
```
In this case, we will include requirements.txt in the docker image, to later access them if we need to. Note that we do not add (`ADD`) code or data since we want to be able to make local changes. 

### Build this *Dockerfile* to get a docker image

builds a docker image from a file

```bash
docker build -f <file-name> -t <image-name> .
```

`-f` file name of the docker. If not defined, it will look for *Dockerfile*

`-t` human readable name of the docker image, which is being created (this is then used to run the container).

`.` tells Docker to look for the docker file in this directory.

### Run a docker container/deamon (for local development)

#### Option 1: Run docker container (interactively)
Run a docker container in interactive mode. This means you can start code within the container that starts. The container will be removed once you exit it (Crtl+D). However, you are not able to browse through the directories within the docker at the same time as you run some code. 

You should mount the code and data directory using `-v`. With this, changes that will be made within the container will be presistend locally (so also when exiting the container). This is handy if you'd like to generate some output or logging, which you'd like to access from outside the container.

```bash
docker run --gpus all -it --rm --ipc=host -v /local_dir/:/container_dir/ --name <container-name> <image-name>
```

Example:
```bash
docker run --gpus all -it --rm --ipc=host -v /mnt/e/Documents/UbuntuCode/GPU_Test/code/:/workspace/code/ --name mypytorchproject pytorch/pytorch:1.4-cuda10.1-cudnn7-devel
```

`--gpus` Useage of available CUDA GPUs (`all`, `0`, `1`, ...)

`-it` run in interactive mode

`--rm` removes the container when finished

`--ipc` use hosts inter-process communication namespace for shared memory. When using torch multiprocessing for multi-threaded data loaders, default shared memory segment size might not be enough. 

`-v /local_dir/:/container_dir/` local_dir is the directory or file from your host system (absolute path) that you want to access from inside your container. Mount your project directory here with code and data to work on it from inside the container at the `/containerdir/` path.

`--name <container-name>` assign name of the container (for future reference). For example, to start container with `docker start <container-name>`

`<image-name>` the name of the image to use as a basis to create the container. 

#### Option 2: Run docker daemon (interactively)
Run a docker daemon in interactive mode. This means you can start code within the deamon that starts. At the same time, you can browse through the dockers file system in a seperate terminal. 

```bash
docker run --gpus all -dit --ipc=host -v /local_dir/:/container_dir/ --name <container-name> <image-name>
```

Then, start a docker deamon to work interactively:
```bash
docker exec -it <container-name> /bin/bash
```

`--gpus` Useage of available CUDA GPUs (`all`, `0`, `1`, ...)

`-dit` detach (a deamon) + run in interactive mode 

`--ipc` use hosts inter-process communication namespace for shared memory. When using torch multiprocessing for multi-threaded data loaders, default shared memory segment size might not be enough. 

`-v /local_dir/:/container_dir/` local_dir is the directory or file from your host system (absolute path) that you want to access from inside your container. Mount your project directory here with code and data to work on it from inside the container at the `/containerdir/` path.

`--name <container-name>` assign name of the container (for future reference). For example, to start container with `docker start <container-name>`

`<image-name>` the name of the image to use as a basis to create the container. 

### Push to dockerhub
https://docs.docker.com/get-started/04_sharing_app/
First, you need to generate an account at [Dockerhub](https://hub.docker.com/).

### Use on cluster 
You should be able to use the docker image on dockerhub for the submit script of slurm on our cluster. It should look somewhat like: 

```bash
submit ./run_script.sh --name test-docker --custom hannahkniesel/myfirstimage --gpus 3090
```
> **_FINAL NOTE:_** To be consistent between the cluster and locally, you should always mount the full directory to the image. As this will happen automatically on the cluster, we can access folder structures in a similar fashion locally and on the cluster.

## More important commands with docker

Show all (locally available) docker images
```bash
docker image ls
```

Show all docker containers
```bash
docker container ls -a
```

Start a docker container
```bash
docker start <container-name>
```

Stop a docker container
```bash
docker stop <container-name>
```

Execute a docker container in interactive mode (can be used to run code from here)
```bash
docker exec -it <container-name> /bin/bash
```

Remove docker container
```bash
docker rm <name>
```


## Optional: Run docker image for releases 
Runs a docker image that has been build from a file that already includes the code. However, since the docker image would become rather big, we do not include the data, but mount the volume via the -v flag.  

```bash
docker run --gpus all --ipc=host -v data:/workspace/data --name <container-name> <image-name>
```

An example *Dockerfile* for release could look somewhat like this: 
```docker
FROM pytorch/pytorch:latest
RUN pip install numpy
ENV CUBLAS_WORKSPACE_CONFIG=:16:8 
ADD code/ /workspace/code/
ENTRYPOINT [ "python", "/workspace/code/main.py"]
```
The code is added to the docker itself. Additionally, the python script will be directly executed on the run of the docker.
However, such a release docker is usually not nessecary in research, as one would like to reproduce multiple experiments or adapt the code in order to fit new needs. 
Furthermore, it still requires the mounting of some output directory for logging. Otherwise all files will only be created *within* the docker container.




