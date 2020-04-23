## Setup

[Install Python 3.8](https://www.python.org/downloads/).

[Install Drake for Python](https://drake.mit.edu/python_bindings.html#inside-virtualenv).
```sh
curl -o drake.tar.gz https://drake-packages.csail.mit.edu/drake/nightly/drake-latest-<platform>.tar.gz
mkdir -p venv
tar -xvzf drake.tar.gz -C venv --strip-components=1
venv/share/drake/setup/install_prereqs
python3 -m virtualenv -p python3 venv --system-site-packages
```

Run virtualenv:
```sh
source venv/bin/activate
```
