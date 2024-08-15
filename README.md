# Discrepancy-aware Probabilistic Meta-Learning for Cold-Start Recommendation

This is our PyTorch implementation for the paper:

> Discrepancy-aware Probabilistic Meta-Learning for Cold-Start Recommendation.



## Highlights

- To enhance the model's ability to capture personalized preferences of cold-start users while maintaining the model's generalization ability, we comprehensively consider intra-user/inter-user consistency/discrepancy, rather than modeling consistency alone.
- To minimize information loss resulting from separately considering consistency and discrepancy, we adopt the principle of contrastive learning to constrain intra-user consistency, intra-user discrepancy and inter-user discrepancy. Inter-user consistency is still achieved through the modeling of distribution over functions and amortized training paradigm.
- Extensive experiments have been conducted on four wildly used benchmark datasets, demonstrating significant improvements over several state-of-the-art baselines.



## Environment Requirement

The code has been tested running under Python 3.7.10. The required packages are as follows:

- pytorch == 1.4.0
- numpy == 1.20.2
- scipy == 1.6.3
- tqdm == 4.60.0
- bottleneck == 1.3.4
- pandas ==1.3.4



## Example to Run the Codes

The parameters have been clearly introduced in `main.py`. 

- Last.FM dataset

  ```
  python main.py --dataset=lastfm --lambda=0.5 --temp=1
  ```

- ML 1M dataset

  ```
  python main.py --dataset=ml1m --lambda=0.5 --temp=1
  ```

- Epinions dataset

  ```
  python main.py --dataset=epinions --lambda=0.1 --temp=0.3
  ```

- Yelp dataset

  ```
  python main.py --dataset=yelp --lambda=1 --temp=1
  ```

  

