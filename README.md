# Laminar Column Model for Winner-Take-All Decision Making

Contains a coupled column class for simulating interaction between two columns. The script `single_sim` performs a single simulation using the parameters specified in the column and simulation configuration files.

From within the root directory, you can run to execute the script.
```python
python -m scripts.single_sim
```

This performs a simulation of two coupled MT columns. If you want to simulate two columns of another region of interest you can pass `region` as an argument when running the script, for example:
```python
python -m scripts.single_sim --region 'v1'
```

In contrast to Kris version of the model, the noise has been removed and the differential equation for the current has been replaced by its steady state (because its dynamics are much faster than those of the membrane potential). The lateral synaptic strength has been adjusted to be in a winner-take-all regime. 
