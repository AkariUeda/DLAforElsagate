
# Summary
* [ Database ](#database)
* [ Software Requirements ](#reqs)
* [ Data Preprocessing ](#preproc)
* [ Frame Extraction ](#frames)

# How to contribute with the project

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

<a name="database"></a>
# Database

The database can be downloaded from the [author's Google Drive](https://drive.google.com/open?id=12nWpZDxhQKC3c9N55F-azefqwgFg5PMl).

Note that you need permission to do so. Send an e-mail to `ueda.aka@gmail.com` so we can authorize your e-mail address to access the data.

In this link, you will find two directories with the training and validation set, in which you will find:
* The Elsagate videos downloaded from YouTube:  All videos from the **training set** were cut to have at maximum 3 minutes and 15 seconds (VERIFICAR). For the **validation set**, we used the full videos to validate the filtering method as it would work in production.
* The extracted and processed frames and motion vectors. The preprocessing steps are explained in the next section.
* The two folds in the training set used in the experiments.

## About Data Annotation
Please note that we did not perform a mannual data annotation. Videos downloaded from official channels (e.g., Disney Channel, Cartoon Network) were considered safe and those downloaded from channels considered Elsagate in the [r/Elsagate subreddit](https://www.reddit.com/r/ElsaGate/comments/6o6baf/what_is_elsagate/) were labeled as Elsagate.

Thus, we are aware that we have some mistakes in the video labeling, mainly in the Elsagate class.  


<a name="reqs"></a>
# Software Requirements

Further explanations on when to use these library or other setup speficiations will be detailed on the next sections.

* Boost
* Caffe
* FFMPEG (Version used: 2.7.2)
* LIBSVM (Version used: 3.18) 
* OpenCV (Version used: 2.4.10)

# Methodology Pipeline

1. **Data Preprocessing**: Extract/Generate low-level data (static and/or motion)
2. Use a Deep Learning Architecture (DLA) model to extract the features from the low-level data
3. Pool the features into a single global  descriptor of the video
4. Predict the class of the video through frames and motion vectors separatedly using SVM
5. Fusion the frames and motion vectors scores to get a final classification



___
# Feature Extraction


