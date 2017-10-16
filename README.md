# Med-dit : Finds the medoid of n points in O(nlog n) steps (naive approach takes O(n^2) steps).

This is a codebase to reproduce all the figures and numbers of the paper titled - "Medoids in Almost-Linear time via
multi-arm bandits".

 1) All the figures can be viewed and generated via ipython notebooks in 'figure' folder
 2) The above figures are generated from experiments - stored in 'experiments' folder
 3) The stored experiments can be re-generated using the following four lines of code

  * python algorithm.py --dataset rnaseq20k --num_exp 1000 --num_jobs 32 --verbose False
  * python algorithm.py --dataset rnaseq100k --num_exp 1000 --num_jobs 32 --verbose False
  * python algorithm.py --dataset netflix20k --num_exp 1000 --num_jobs 32 --verbose False
  * python algorithm.py --dataset netflix100k --num_exp 1000 --num_jobs 32 --verbose False

    * dataset - name of the dataset
    * num_exp - Number of total experiments
    * num_jobs - Number of experiments run parallely
  
