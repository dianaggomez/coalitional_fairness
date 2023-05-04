# coalitional_fairness

Coalition Fairness analyzes fairness of actions between groups of Autonomous Vehicles that act selfishly and the traffic effects of a fairness regulation.

## Installation


Create virtual environment
```bash
conda create -n coalitional_fairness python=3.7
conda activate coalitional_fairness 
```
Create directory
```bash
mkdir coalitional_fairness
cd coalitional_fairness
```

Clone repository
```bash
git clone https://github.com/dianaggomez/coalitional_fairness.git

```
Install the multiagent High Level Controller environment for training

```bash
cd coalitional_fairness/mutliagent
pip install -e .

cd ..
cd ..
```

Download Copo to use ippo algorithm

```bash
git clone https://github.com/decisionforce/CoPO
cd CoPO/copo_code
pip install -e .

```

Move the following files

```bash
cd 
cd coalitional_fairness
move ~/coalitional_fairness/ippo/train_hlc_ippo.py ~/CoPO/copo_code/copo
move ~/coalitional_fairness/ippo/callbacks.py ~/CoPO/copo_code/copo
```

Folder structure:

    coalitional_fairness
    ├── coalitional_fairness                   
    │   ├── multiagent         
    │   ├── ippo         
    │   └── ...               
    └── CoPO
    
## Usage

To train the High Level Controller environment using CoPo ippo algorithm
```bash
cd CoPO/copo_code/copo/
python train_hlc_ippo.py --exp-name hlc_ippo --num-gpus=NUM_GPUS
```


## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.
