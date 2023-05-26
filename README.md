# Kinetic Soundscapes

Software program described in this paper: https://digitalcommons.dartmouth.edu/masters_theses/63/

The motion synthesizer python program is powered by FoxDot, which in turn runs on SuperCollider.
To get set up with SuperCollider and FoxDot (and Python if you don't already have it), follow [this guide](https://foxdotcode.readthedocs.io/en/latest/guides/installation.html). Be sure to follow the instructions for "Installing SC3 PLugins" as well, as some of those additional features are used in the program.

To install required dependencies and activate the virtual environment for the python program, run
```
source setup.sh
```

To start having SuperCollider listen to signals from FoxDot, open SuperCollider and evaluate `FoxDot.start`. This must be always be done prior to running the motion synthesizer, as SuperCollider is the underlying music engine and FoxDot is a Python library that sends signals from the motion synthesizer program to SuperCollider.

To configure FoxDot to use SC3 Plugins (this only needs to be done once on your machine), start FoxDot from within the virtual environment, and check Language > Use SC3 Plugins, then close out of FoxDot.
![241128990-ea536cac-5796-4d01-894b-38a40eacefc4](https://github.com/hzwus/kinetic-soundscapes/assets/56451162/59e63125-7dc5-478e-b977-f6b3bda3ccc5)


Now you can run the program with
```
python motion_synthesizer.py
```
