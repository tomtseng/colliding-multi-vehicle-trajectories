# Multi-vehicle trajectory planning with collisions

Final project for [MIT's Spring 2020 iteration of 6.832: Underactuated
Robotics](http://underactuated.csail.mit.edu/Spring2020/). See `writeup.pdf` for
details.

## Setup

[Install Drake for
Python](https://drake.mit.edu/python_bindings.html#inside-virtualenv). This
requires Python 3.8 for macOS users and Python 3.6 for Ubuntu 18.04 users.
Commands:
```sh
curl -o drake.tar.gz https://drake-packages.csail.mit.edu/drake/nightly/drake-latest-<platform>.tar.gz
mkdir -p venv
tar -xvzf drake.tar.gz -C venv --strip-components=1
sudo venv/share/drake/setup/install_prereqs
python3 -m virtualenv -p python3 venv --system-site-packages
```
where [`<platform>` is either `mac` for macOS or `bionic` for Ubuntu 18.04](https://drake.mit.edu/from_binary.html#binary-installation).

Run virtualenv:
```sh
source venv/bin/activate
pip install -r requirements.txt
```

## Run the animation

To produce an animation, run the following command:
```sh
python run.py
```

See the file comment on run.py to change the vehicle start and stop locations.
