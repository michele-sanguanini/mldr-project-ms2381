# Machine Learning Real Data extra project

Let's setup this project to start working with data science projects as if we were in a company.

## Requirements

This project will run in a containerised manner in order to increase compatibility across machines (and not to leave trace on your system once we are done with the project). So you will need to install docker. A good way is to use the cross-platform open source software Rancher Desktop: https://github.com/rancher-sandbox/rancher-desktop/releases

## First setup 

In order to start the project you need to have the docker daemon up and running (for instance via opening Rancher Desktop and waiting until it's up and running).

We then want to build the container via running
`docker compose up -d --build`
this will create a detached docker instance that runs various containers that we need for this project (namely: ollama, minIO, our main app container). The first time you will run this command a few images will be downloaded onto your machine.

After this is done you can open a shell in the Python/uv/DVC container:
`docker compose exec app bash`

This will turn your terminal into something like:
` root@66014c2d803e:/workspace# `
and you can exit it via typing `exit` within the app terminal.

While within the shell, try and type 

`dvc version`

`uv venv; uv pip install ollama`

From your browser try to go to the following addresses:

minIO web console: http://localhost:9001/

ollama host: http://localhost:11434 

Try standing down the containers using `docker compose down` and then go to the addresses again. Run `docker compose up` again to restart the containers

### Let's try to run ollama from the container

So far we have:
- started a local service that runs ollama in a container independent from your machine;
- made available a lightweight LLM model (phi3);
- setup python.

Let's try and check that we can access the phi3 model from within our app:
- let's open the app shell `docker compose exec app bash`
- let's install the ollama python client if we haven't done that already: `uv venv; uv pip install ollama`
- let's open python `uv run python`
- this is the script we want to run by copy-pasting:
```
import ollama

response = ollama.chat(
    model="phi3",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response["message"]["content"])
```
- the result should be a generated AI response; in case of an error like `ollama._types.ResponseError: model requires more system memory (3.5 GiB) than is available (2.7 GiB) (status code: 500)` you might need to allocate more memory to the virtual machine (in Rancher Desktop it is sufficient to go to `Preferences > Virtual Machine > Memory (GB)` and set the limit to 6GB);
- once you stand down the container (unless you have some local uv/ollama setup) you should not be able to do any of the actions you've just taken.

# Model tracking with DVC

In order to setup some basic ML experiment tracking using DVC we want to 
