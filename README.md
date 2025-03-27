## CS584 Final Project: Note Filer
The objective of this project is to be able to take notes and classify them automatically using a language model. \
TBD 


## How To Use
First, clone the repository.

```
git clone git@github.com:brrryry/note-filter.git
```

### Using The Bot
Enter the bot directory.

```
cd bot
```


Install dependencies:

```
npm i
```

Modify the ``.env`` file to tailor your discord application. \
From there, you should be all set! You can run ``npm deploy-commands.js`` to deploy slash commands, and you can use ``node .`` to start the bot up! 

### Starting The Flask API
Enter the model directory.

```
cd model
```

(Optional) Create a Python Virtual Environment.

```
python -m venv .venv
```
Activate your environment (for Windows):
```
./.venv/bin/Activate
```
Activate your environment (for Linux):
```
source ./.venv/bin/activate
```
Install the requirements!
```
pip install -r requirements.txt
```
Run the flask app:
```
flask --app main run
```

**NOTE: You need to run both of these programs simultaneously for the project to work!!** 

That's all! Happy programming :)