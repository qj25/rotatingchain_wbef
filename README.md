# Robotic manipulation of a rotating chain with bottom end fixed

This paper studies the problem of using a robot arm to manipulate a uniformly rotating chain with bottom end fixed. We find that the configuration space of such a chain is homeomorphic to a three-dimensional cube. Using this property, we suggest a strategy to manipulate the chain into different configurations, specifically from one mode to another, while taking stability and feasibility into consideration. We demonstrate the effectiveness of our strategy in physical experiments by successfully transitioning from rest to the first two rotation modes. We discuss how the concepts explored in our work has potential applications in the manipulation of drill strings and the yarn spinning process.

Authors: Qi Jing Chen, Shilin Shan, Quang-Cuong Pham

## Required
Install [`eigen`](http://eigen.tuxfamily.org/index.php?title=Main_Page#Download) package in `home` root directory.

## Usage
Note: All data and figures saved will be located in 'data' folder.
1. In the root directory of this package, run:
```
pip install -e .
```
2. Run:
```
cd rotatingchain_wbef/algos/shootingc_cpp
bash swigbuild.sh
cd ../../..
cd rotatingchain_wbef/stability/stabjac_cpp
bash swigbuild.sh
cd ../../..
```
3. To obtain configuration data of chains:
```
cd scripts
python shootv2c.py 3d
python shootv2c.py stab
python shootv2c.py c
python stability_testerb.py
```

4. In 'scripts',
- plot Fig.2 (B):
```
python show_final_path.py shapes
```
- plot Fig.4:
```
python data_plotter2b.py
```
- plot Fig.5:
```
python show_final_path.py full
```
- plot Fig.6:
```
python detailed_stab.py 
```
- plot Fig.7 (A):
```
python show_final_path.py path
```
- plot Fig.7 (B) -- does not save immediately, only plots:
```
python show_final_path.py controls
```
