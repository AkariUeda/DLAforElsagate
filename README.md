Code to reproduce the results for the paper ["Combating the Elsagate phenomenon: Deep learning architectures for disturbing cartoons"](https://arxiv.org/pdf/1904.08910.pdf) in 7th IAPR/IEEE International Workshop on Biometrics and Forensics (IWBF).

# Summary
* [ Database ](#database)
* [ Methodology Pipeline ](#reqs)
* [ Data Preprocessing ](https://github.com/AkariUeda/DLAforElsagate/tree/master/data_preprocessing)

<a name="database"></a>
# Dataset

You can download the data from the [author's Google Drive](https://drive.google.com/open?id=12nWpZDxhQKC3c9N55F-azefqwgFg5PMl).
It is necessary to sign a license agreement to get access to the data, you can find the license agreement [here](). Please sign it and send a copy to Sandra Avila <sandra [at] ic [dot] unicamp [dot] br>.

In this link, you will find two directories with the training and validation set, in which you will find:
* The Elsagate videos downloaded from YouTube:  All videos from the **training set** were cut to have at maximum 3 minutes and 15 seconds. For the **validation set**, we used the full videos to validate the filtering method as it would work in production.
* The extracted and processed frames and motion vectors. The preprocessing steps are explained in the next section.
* The two folds in the training set used in the experiments.

### About Data Annotation
Please note that we did not perform a manual data annotation. Videos downloaded from official channels (e.g., Disney Channel, Cartoon Network) were considered safe and those downloaded from channels considered Elsagate in the [r/Elsagate subreddit](https://www.reddit.com/r/ElsaGate/comments/6o6baf/what_is_elsagate/) were labeled as Elsagate.

# Methodology Pipeline

Access specific instructions for each step of the pipeline through the links below:

1. [**Data Preprocessing**](https://github.com/AkariUeda/DLAforElsagate/tree/master/data_preprocessing): Extract/Generate low-level data (static and/or motion)
2. [**Models**](https://github.com/AkariUeda/DLAforElsagate/tree/master/weights): Choose a model to execute a feature extraction or finetuning.
3. [**Feature Extraction**](https://github.com/AkariUeda/DLAforElsagate/tree/master/feature_extraction): Use a Deep Learning Architecture (DLA) model to extract the features from the low-level data and pool the features into a single global descriptor of the video.
4. [**Classification**](https://github.com/AkariUeda/DLAforElsagate/tree/master/classification): Predict the class of the video through frames and motion vectors individually using SVM and fusion the frames and motion vectors scores to get a final classification.

# Citation

If this work/repository was useful for your project, please consider citing our paper.

```
@inproceedings{ishikawa2019dlaelsagate,
 title={Combating the {Elsagate} Phenomenon: {D}eep Learning Architectures for Disturbing Cartoons},
 author={Akari Ishikawa and Edson Bollis and Sandra Avila},
 booktitle={7th IAPR/IEEE International Workshop on Biometrics and Forensics (IWBF)},
 year={2019}
}
```

Also, our work was largely based on Mauricio Perez work: [Video pornography detection through deep learning techniques and motion information](https://www.sciencedirect.com/science/article/pii/S0925231216314928). We strongly recommend the reading if you are planning on reproducing our experiments.


# Acknowledgments

* A. Ishikawa is funded by PIBIC/CNPq, FAEPEX (\#2555/18) and Movile. 
* E. Bollis is funded by CAPES. 
* S. Avila is partially funded by Google Research Awards for Latin America 2018, FAPESP (\#2017/16246-0) and FAEPEX (\#3125/17).
* RECOD Lab. is partially supported by diverse projects and grants from FAPESP, CNPq, and CAPES. 
* We gratefully acknowledge the support of NVIDIA Corporation with the donation of the Titan Xp GPUs used for this research.
