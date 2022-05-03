## A simple package manager for Ryven

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
## Installation
Make a new folder somewhere on your device
Drag and drop `package-loader.py` and `settings.json` into this new folder
Add a new folder inside, titled `packages`
On the project repository page, several modules are included under the `packages`folder. Drag whatever modules you will need into your `packages` folder.


