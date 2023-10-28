
Forked from https://github.com/openai/shap-e
### Run API

```shell
# Run this command from a shell at shap-e to start the api.
uvicorn main:app --reload

#euler
sbatch .start

#copy file from euler
scp <username>@euler.ethz.ch:<path> .
```

Endpoint will be available at http://localhost:8000/shape

there are 2 query parameters, prompt & nr_samples

prompt accepts a string prompt to give to the model.

nr_samples is optional & defines the number of samples the model will create for the prompt.
default is 3