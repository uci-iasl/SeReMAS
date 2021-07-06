# SeReMAS
This repository complements the paper **SeReMAS: Self-Resilient Mobile Autonomous Systems Through Predictive Edge Computing** accepted for publication at IEEE International Conference on Sensing, Communication and Networking 2021.
[preprint](https://arxiv.org/abs/2105.15105)
conference - to come when published on IEEE
```bibtex
@article{callegaro2021seremas,
  title={SeReMAS: Self-Resilient Mobile AutonomousSystems Through Predictive Edge Computing},
  author={Callegaro, Davide and Levorato, Marco and Restuccia, Francesco},
  journal={arXiv preprint arXiv:2105.15105},
  year={2021}
}
```

[![Conference video teaser](https://i.ytimg.com/vi/Tv1cW5uag-8/hqdefault.jpg?sqp=-oaymwEcCNACELwBSFXyq4qpAw4IARUAAIhCGAFwAcABBg==&rs=AOn4CLDTbZjePPSVWaSjJ_w2jwzdpDvGAA)](https://www.youtube.com/watch?v=Tv1cW5uag-8&ab_channel=IEEESECONConference)


### Installation instructions:
- clone repository
```bash
git clone https://github.com/uci-iasl/SeReMAS.git
```
- enter the directory and create a Python3 virtual environment and install the requirements into the newly created virtual environment
```bash
cd SeReMAS
python3 -m venv venv
source venv/bin/activate
pip install -r requirements
```
- (install and) run a [SITL](https://ardupilot.org/dev/docs/setting-up-sitl-on-linux.html)
```bash
cd ardupilot/ArduCopter
sim_vehicle.py
```

### Fast run
Now we want to run 2 scripts: on is an edge server, which supports the computation of the drone.
You can run the edge server:
```bash
python edge.py localhost
```
Then run the drone script:
```bash
python drone_demo.py --fly --move --c 127.0.0.1:14550 --model 2
```

### Add some visual
Note: if you want to see the drone moving in some sort of visual manner, I recommend the following setup.
- Download [SITL](https://ardupilot.org/dev/docs/setting-up-sitl-on-linux.html)
- When running the SITL simulator
```bash
cd ardupilot/ArduCopter
sim_vehicle.py
```
in the pop up terminal use
```bash
output add 127.0.0.1:14553
```
This allows us to connect two instances to the simulated copter. One is the visualization, the other the SeReMAS framework.
- Connect QGroundControl - this will connect automatically to 127.0.0.1:14550. If it does not happen, the GUI is easy to use. Send me an email if it takes more than 15mins!

Now run SeReMAS:
```bash
python drone_demo.py --fly --move --c 127.0.0.1:14553 --model 2
```
You notice we changed the port we connect to. That's it, that's the magic.

### Suggestions
Use this software mainly for flying simualted MAVLink enabled environment.
Note that when using it on a real Unmanned Aerial Vehicle we always respect all local, regional and national regulations. We always have a strong reliable Radio Controlled connection to Return To Launch or Land or turn off if necessary. Multi-copters are inherently dangerous and their operation should be done with plenty of caution.

### Disclaimer
All the code provided is given as-is. This is prototype-grade code, not extensively tested. Run at your own risk.
We are not responsible for any improper use. Strongly suggest to use it only in simulated enviroment. 

Let me know if I can help with an email to [dcallega@uci.edu](mailto:dcallega@uci.edu).
