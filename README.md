## A simple package manager for Ryven
# NOTICE: This project is NOT complete and has been halted in progress, however progress may be made in the future



This package manager includes a Python script that will automatically import several code libraries. Each subfolder under the  `packages`folder contains all of the integration code of the associated module:

    root/
    ├─ packages/
    │ ├─ math_module/
    │ │ ├─ nodes.py
    │ │ ├─ widgets.py
    │ ├─ .../
    │ │ ├─ nodes.py
    │ │ ├─ widgets.py
    package-loader.py
    settings.json
Instead of importing several libraries manually, simply run `package-loader.py` to import packages such as `math_module` directly into Ryven. Use`settings.jsonc` to select which modules to import, and also whether or not to import them all in unison, or one by one.

Even if you don't feel it's necessary to use `package-loader.py`, several modules like the the built-in library `math` are available so there is no need to port the packages over yourself.

The idea of this repository is to provide modularity, and as such, all of the packages, such as `math-module`are optional

##Tools
The folder `tools` contains several tools, mostly for development purposes. One such tool is in `code_gen.py`, and as you might have guessed, it takes a function and ports it into Ryven. This tool is purely experimental and as such, bugs are to be expected. Consider it a code *template* creator, not a finished module.


    Usage:
    Let's say you have some python module you want implemented into Ryven as nodes.
    Right now, this project is not a python package, so you cannot import any of these tools into any python file - You must execute it within the Ryven-Package-Manager.
    
    In order to get a template up and running, simply import that module and use the `Ryven_Nodifier.nodify` method on the function / method in question. You will have to call each method, one at a time at the moment, but there is currently plans for development on `Ryven_Nodifier.nodify_module`, a method that will nodify an entire module's attributes, recursively.
    
In `param_counter.py`, the method `get_params` is actually a super - useful function for introspection, since it does have limited functionality on some C-bound functions that normally cause errors with `inspect.getargspec` and the like.
`get_params` is a modification and evolution of this function:
https://stackoverflow.com/questions/48567935/get-parameterarg-count-of-builtin-functions-in-python
## Installation
Make a new folder somewhere on your device
Drag and drop `package-loader.py` and `settings.json` into this new folder
Add a new folder inside, titled `packages`
On the project repository page, several modules are included under the `packages`folder. Drag whatever modules you will need into your `packages` folder.


