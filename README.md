## Code to paper submission


#### 1. We require PU model's prediction and matrix priors to estimate noise transition matrices, which both take some time to compute. Some example data (prior matrices and retrain csv data) is provided, please note that the prior file and retrain file should match with the noise rate and noise type. For example, prior './avg_priors/avgp_initialm_intactpw20.npy', retrain file './retrains/cifar10retrain_intactpw20.csv' match with noise rate 0.2 with noise type pw (pairwise). For more examples, please find the example data in the given prior directory './example_data/avg_priors' and retrain file directory './example_data/retrains'. An example run would be:

An example: <br/>
```
python main.py --noise_rate 0.2 --form pw --prior_path ./example_data/avg_priors/avgp_initialm_intactpw20.npy --retrain_file_path ./example_data/retrains/cifar10retrain_intactpw20.csv
```

`--noise_rate`: noise rate in [0, 1], <br/>
`--form`: pw (pairwise) or sym (symmetric) noise type <br/> 
`--prior_path`: path to load precomputed priors, the default directory is ./avg_priors <br/> 
`--retrain_file_path`: path to load precomputed PU model predictions, the default directory is ./retrains <br/>


#### 2.1 To generate your own prior matrices, just leave --prior_path blank and indicate the noise rate, noise type and available GPU for training neural network to get prior:

An example: <br/>

```
CUDA_VISIBLE_DEVICES=0 python main.py --noise_rate 0.2 --form pw --retrain_file_path ./example_data/retrains/cifar10retrain_intactpw20.csv
```

#### 2.2 To generate your own retrain csv, please follow below command:

##### 2.2.1 First, generate LID sequences. There are uniform alpha vectors and 10 random seeds pre-set in the code. The preprocessed cifar-10 data can be found in the experiment section in https://github.com/xiaoboxia/T-Revision. Running below example would generate LID sequences from the pre-set alpha vectors and the 10 random seeds for a noisy dataset with pairwise noise rate 0.2:
An example: <br/>
```
CUDA_VISIBLE_DEVICES=0 python LID_generation.py --form pw --noise_rate 0.2
```

##### 2.2.2 After the LID sequences are obtained, train PU models for the first time (the LID sequences are saved in a predefined directory './log') and retrain them. Note that below code also has pre-set random seeds, which should be the same as the ones in above command of 2.2.1. Then the retrain csv is obtained by:

An example: <br/>

```
python initial_and_retrain.py 
```

##### Note that the randomness of running the code is to be minimized so that it would only be affected by the random seed for LID generation, i.e., run the same code twice, the LID sequences would be exactly the same. However, even running the code on the same GPU clusters, if the number of GPUs used are different, i.e., 1 vs 2, the LID sequences would be different. Therefore, to allow LID sequences reproducible, it is recommended to run the code on the same (type and number of) device.


#### Requirements:
pytorch, numpy, scipy, sklearn, pandas, csv


