# **Introductory information**

## Data Mining NLP python project:
**The goal** was to collect the data, build the Neural Network for text binary classification using only _Pytorch, 
<br>Tensorflow, Pandas, Numpy and BeautifulSoup_. The implementation was be done in _Python 3.6+._ 
<br>For web scraping was used _Grequests_ Python library

It contains of 2 parts: 
- Scraping data from recipe website part (data collecting)
- Data Science part (it made around the classification problem that determines for each paragraph with what probability it’s 
  <br>label is ‘ingredients’ or ‘recipe’)
#### !!!These parts are not running as one pipeline!!! No console output - only log (except when you run run.sh)!!!

<br>All functions are wraped with `@profile` decorator - it prints out into log file time and memory usage during code 
<br>running program execution. The code of this utility you may find in **utils.py** in the main project directory.
<br>The project has the only one **recipes_logging.log** file, as well in the main directory. The project has **run.sh**
<br>file, which you run in cli  `bash run.sh htth://www.anylinkfromtestlinksfileindatafolder.com`. As the result you 
<br>will get  in the terminal window a json-like text from entered link page. In the **recipes_logging.log** file will 
<br>be information about all functions ran and steps were made, classification, probability and metrics. 
<br>In **requirements.txt** you can find applied Python packages.
<br>In **notebooks_and_drafts** just notebooks and drafts. The project run doesn't touch it.
<br>In **config.py** all constants (or almost all)
## Methodological information
### **Scraping data from recipes website**

Unique links for scraping are in **data/all_recipes_links.txt**
<br>Links without needed content/class-names are in **data/no_recipe_page.txt**
<br>Scraped data {'Recepie':[text],'INGREDIENTS': text} is in **data/recipes.pkl**
<br>The scraping part starts from **main_scraper.py**. It extracts all links from a page with all recipes using 
<br>**extract_all_links.py**,writes down collected links into **data/all_links.txt** file, redact the all_links.txt --> 
<br>leaves only links with no duplicates and pages with relevant data. Then reads links from a file and calls 
<br>**extract_one_recipe.py** to collect data from each recipe page. Saves collected data into **data/recipes.pkl** file.
<br> If you want to continue run the project, then run **preprocess.py**

<br>_**IMPORTANT:**_ If you want to take out several links to test the model, you have to uncomment the line 34, 
<br> in **main_scraper.py** before run it (to stop the program run in the middle) and in hand way take out links from 
<br> **data/all_links.txt**, then save the file and press enter in terminal. In such way data from these pages won't be 
<br>collected into data set for training the model.
<br>If you want just to test the model work using **run.sh** file, then you can use already written links in **data/test_links.txt**
### **Preprocessing, modeling, feature engineering**
Next stage is starting from run `main_preprocess(filename)` in **preprocess.py**.This function load the data after 
<br>scrapping from **data/recipes.pkl** file. Calling for `load_data_transform_to_set(filename)` to transform into data 
<br>set with column `paragraph` and `label`,then call for `utils.stratified_split_data(text, label, TEST_SIZE)` and 
<br>after that preprocess separately train/test sets using `preprocess_clean_data(train_dataset.as_numpy_iterator(), f'train')`.
<br>In  `preprocess_clean_data` function clean the series of text and add new columns with additional features. New 
<br>sets save into 2 pkl files: **data/train_data_clean.pkl** and **data/test_data_clean.pkl**.
<br>This is the end of first Data Science part in this project. Next step is in **model_train.py**. After this step we 
<br>can start to tune the model. This was the reason why preprocessing part is finished exactly here.

### **Train, tune, evaluation**
<br>When you run **model_train.py**, at first it will read  just saved on the previous step **data/train_data_clean.pkl** 
<br>and **data/test_data_clean.pkl**. Then count `max len sent/sequence` and `vocabulary size`. Next will be call for 
<br>`preprocess.tfidf(texts, vocab_size) ` function to transform data into sequences. Then split data into nlp and meta 
<br>sets for test and train sets, call for
        `preprocess.get_model(tf_idf_train, X_meta_train, results, embedding_dimensions=EMBEDDING_DIM)`
<br> to create, build and train the MODEL. Next step is evaluation on the test sets with writing down results into log 
file, plot loss vs val_loss, save the model into `config.MODEL_NAME='data/my_model.h5'`. Here is the end of the modeling
<br>and preprocessing.

And **_the last final stage:_** to run in terminal `bash run.sh htth://www.anylinkfromtestlinksfileindatafolder.com`. 
It will call **main_task_run.py** which accepts arguments such as your website link. Then it will call for 
<br>`extract_one_recipe.get_recipe(url_to_get)` from the scraping part to collect needed data from the page. We are
<br>assuming  that url is valid and that it redirects you to a page where a valid recipe is located. Then it calls the 
<br>`utils.print_json(url_to_get_recipe, json_file)` function which give you console output in json-like format.
<br>Next, apply transformation for first part of the json file: from `list` ==> to `string`. Call the function 
<br>`preprocess.load_data_transform_to_set()` to transform from a dictionary into data set. Preprocess text, engineer 
<br> new features, save to pikle file using `preprocess.preprocess_clean_data()`. Split into nlp and meta sets and call
<br> `eval_on_one_page(tfidf_one_page, X_meta_one_page, y_one_page, model, text)` from **run_tensorflow.py**. In log file
<br> you will find all information about the metrics, model, table with predictions and probability values.

<br>_**IMPORTANT:**_ If you want to test models in **/data** folder on test links, then you have to run **notebooks_and_drafts/list_dir.py**
<br> In **/data** folder can be several pretrained models. It runs on links from **data/test_links.txt**

## Data specific information

The data is collected from 'https://www.loveandlemons.com'
<br>Data is imbalanced 80/20. 
<br>Paragraph = all lines in ingredients labeled as 1, 
<br>Paragraph = each `\n` in Instructions labeled as 0.
<br>Vocabulary size = 2263 words/lemmas. 
<br>Test split size = 0.2
<br>Data from 10 links wasn't included into set (was using for testing **run.sh**) it is in **data/test_links.txt**