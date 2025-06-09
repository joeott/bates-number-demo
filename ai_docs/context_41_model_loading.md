Manage Models in Memory

AI models are huge. It can take a while to load them into memory. LM Studio's SDK allows you to precisely control this process.

Model namespaces:

LLMs are accessed through the client.llm namespace
Embedding models are accessed through the client.embedding namespace
lmstudio.llm is equivalent to client.llm.model on the default client
lmstudio.embedding_model is equivalent to client.embedding.model on the default client
Most commonly:

Use .model() to get any currently loaded model
Use .model("model-key") to use a specific model
Advanced (manual model management):

Use .load_new_instance("model-key") to load a new instance of a model
Use .unload("model-key") or model_handle.unload() to unload a model from memory
Get the Current Model with .model()
If you already have a model loaded in LM Studio (either via the GUI or lms load), you can use it by calling .model() without any arguments.

Python (convenience API)Python (scoped resource API)
import lmstudio as lms

model = lms.llm()

Get a Specific Model with .model("model-key")
If you want to use a specific model, you can provide the model key as an argument to .model().

Get if Loaded, or Load if not
Calling .model("model-key") will load the model if it's not already loaded, or return the existing instance if it is.

Python (convenience API)Python (scoped resource API)
import lmstudio as lms

model = lms.llm("llama-3.2-1b-instruct")

Load a New Instance of a Model with .load_new_instance()
Use load_new_instance() to load a new instance of a model, even if one already exists. This allows you to have multiple instances of the same or different models loaded at the same time.

Python (convenience API)Python (scoped resource API)
import lmstudio as lms

client = lms.get_default_client()
llama = client.llm.load_new_instance("llama-3.2-1b-instruct")
another_llama = client.llm.load_new_instance("llama-3.2-1b-instruct", "second-llama")

Note about Instance Identifiers
If you provide an instance identifier that already exists, the server will throw an error. So if you don't really care, it's safer to not provide an identifier, in which case the server will generate one for you. You can always check in the server tab in LM Studio, too!

Unload a Model from Memory with .unload()
Once you no longer need a model, you can unload it by simply calling unload() on its handle.

Python (convenience API)Python (scoped resource API)
import lmstudio as lms

model = lms.llm()
model.unload()

Set Custom Load Config Parameters
You can also specify the same load-time configuration options when loading a model, such as Context Length and GPU offload.

See load-time configuration for more.

Set an Auto Unload Timer (TTL)
You can specify a time to live for a model you load, which is the idle time (in seconds) after the last request until the model unloads. See Idle TTL for more on this.

Pro Tip
If you specify a TTL to model(), it will only apply if model() loads a new instance, and will not retroactively change the TTL of an existing instance.

PythonPython (with scoped resources)
import lmstudio as lms

llama = lms.llm("llama-3.2-1b-instruct", ttl=3600)

