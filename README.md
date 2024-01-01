# H&M Personalized Fashion Recommendation


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation-and-usage">Installation and Usage</a></li>
      </ul>
    </li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This project is about H&M Persionalized Fashion Recommendation. You can find it in Kagge with the same name [here](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)

H&M Group is a family of brands and businesses with 53 online markets and approximately 4,850 stores. Our online store offers shoppers an extensive selection of products to browse through. But with too many choices, customers might not quickly find what interests them or what they are looking for, and ultimately, they might not make a purchase. To enhance the shopping experience, product recommendations are key. More importantly, helping customers make the right choices also has a positive implications for sustainability, as it reduces returns, and thereby minimizes emissions from transportation.

In this competition, H&M Group invites you to develop product recommendations based on data from previous transactions, as well as from customer and product meta data. The available meta data spans from simple data, such as garment type and customer age, to text data from product descriptions, to image data from garment images.

| Model | Full name | Paper |
| NAML | Neural News Recommendation with Attentive Multi-View Learning | https://arxiv.org/abs/1907.05576 |

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

1. **Python**: Python is mainly programming language used in this project.
2. **Dataset**: You need to download the H&M dataset from Kaggle. You do not need to download image, as it will not be used.

### Installation and Usage

1. Clone the repo
   ```
   https://github.com/Anh26535D/HnM-personalized-fashion-recommendation
   cd HnM-personalized-fashion-recommendation
   ```
2. Make data folders structures. You need to ensure that the data folders in HnM-personalized-fashion-recommendation.

    ```
    |-- data/
    |
    |-- processed_data/
    |   |-- train/
    |   |-- val/
    |   |-- glove/
    |
    |-- src/
    |   |-- model/
    |
    |-- .gitignore
    |-- README.md
    |-- requirements.txt
    |-- split_train_val.py
    ```

3. Split train val
    ```
    python split_train_val.py
    ```

4. Preprocess data
    ```
    python .\src\data_preprocess.py
    ```

5. Training
    ```
    python .\src\train.py
    ```

6. Evaluation
    ```
    python .\evaluate.py
    ```
