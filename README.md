# Laminar Column Model for Bistable Perception

This repository provides the dynamic mean field implementation of the Laminar Column Model described in:
Layered Structure of Cortex Explains Reversal Dynamics in Bistable Perception ... [full reference once available]

## Abstract
Bistable perception involves the spontaneous alternation between two exclusive interpretations of a single stimulus. Previous research has suggested that this perceptual phenomenon results from winnerless dynamics in the cortex. Indeed, winnerless dynamics can explain many key behavioral characteristics of bistable perception. However, winnerless dynamics fails to explain an increase in alternation rate that is typically observed in response to increased stimulus drive. Instead, winnerless competition predicts a decline in alternation rate with increased stimulus drive. To reconcile this discrepancy, several lines of work have augmented winnerless dynamics with additional processes such as global gain control, input suppression, and release mechanisms. However, the cortical implementation and biological substrates of these mechanisms remain speculative. These offer potential explanations for an increase in alternation rate when stimulus drive increases at an algorithmic level. But it remains unclear which, if any, of these mechanisms are implemented in the cortex and what their biological substrates might be. We show that the answers to these questions lie within the architecture of the cortical microcircuit. Utilizing a dynamic mean field approach, we implement a laminar columnar model with empirically derived interlaminar connectivity. By coupling two such circuits such that they exhibit competition, we are able to produce winnerless dynamics reflective of bistable perception. Within our model, we identify two mechanisms through which the layered structure of the cortex gives rise to increased alternation rate in response to increased stimulus drive. First, deep layers act to inhibit the upper layers, thereby reducing the attractor depth and increasing the alternation rate. Second, granular layer L4's external inputs, depending on their intensity, exhibited a dichotomy in their effects on the alternation rate: weak inputs reduced it, whereas strong ones increased it. This phenomenon was attributed to the recurrent feedback between the superficial L23 and granular L4 layers, which we identified as an input suppression mechanism. These results underscore the importance of interlaminar columnar connectivity. The implications extend to reevaluating the impacts of layer-specific external inputs from bottom-up, top-down, and lateral sources, providing a deeper understanding of the neural underpinnings of bistable perception.

## Hands-On Simulations

### `trial.ipynb`:
This notebook provides an interactive environment for understanding and simulating the Laminar Column Model. The key sections of the notebook are:

1. **Initial Setup**: Import necessary libraries and modules and configure the notebook for inline visualizations.
2. **Model Parameters & Simulation Setup**: Define the parameters for the simulation, initialize model states, and configure external stimulus targeting specific cortical layers.
3. **Model Simulation**: Run the model over time, dynamically visualizing the rate of neural activity and dominance duration distribution at specific intervals.
4. **Bistable Statistics Computation**: Analyze the simulation's outcome, calculating the dominance duration and alternation rate, and presenting the results to the user.

The notebook serves as a hands-on tool for researchers and enthusiasts looking to delve deeper into the dynamic mean field implementation of bistable perception.

### Installation

Ensure you have **Python 3.8** or higher.

**Dependencies:** Install necessary packages using:

```bash
pip install -r requirements.txt
```