# coalitional_fairness

Coalition Fairness analyzes fairness of actions between groups of Autonomous Vehicles that act selfishly and the traffic effects of a fairness regulation.

## Installation



Create virtual environment
```bash
conda create -n coalitional_fairness python=3.7
conda activate coalitional_fairness 
```
Clone repository
```bash
git clone https://github.com/dianaggomez/coalitional_fairness.git

```
Install the multiagent High Level Controller environment for training

```bash
cd coalitional_fairness/mutliagent
pip install -e .

```

Download Copo to use ippo algorithm

```bash
git clone https://github.com/decisionforce/CoPO
cd CoPO/copo_code
pip install -e .

```

Move the following files

```bash
cd..
move ~/ippo/train_hlc_ippo.py ~/CoPO/copo_code/copo
move ~/ippo/callbacks.py ~/CoPO/copo_code/copo
```


## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.
