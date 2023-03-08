# ThePhylogeneticGame
Git repository for the paper : "The tree reconstruction game: phylogenetic
reconstruction using reinforcement learning" <br />
<Link to paper>


In order to install the necessary environment please make sure you have 
Anaconda. Create the environment with <br />

`conda env create -f environment.yml`

Please note this environment may only be installed on a linux machine as RaxML-NG program requires a linux machine.

To run an experiment run the ProjectMain.py file. <br />
There are the relevant parameters: <br />
-- experiment_unique_dir_name - which will create a unique directory for you experiment results.<br/>
-- cpus - which allows you to run the training process on multiple cpus for increased runtime.

All parameters for experiment configuration are in SharedConsts.py with appropriate documentation.