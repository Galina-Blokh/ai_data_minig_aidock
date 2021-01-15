# **Introductory information**

## Data Mining NLP python project:
**The goal** was to collect the data, build the Neural Network for text binary classification using only Tensorflow, 
<br>Pandas, Numpy and BeautifulSoup_. The implementation is in _Python 3.6+._ For web scraping used _Grequests_ 
<br>Python library

It contains two parts: 
- Scraping data from recipe website part (data collecting)
- Data Science part (it made around the classification problem that determines for each paragraph with what probability
<br>it’s the label is ‘ingredients’ or ‘recipe’)
#### !!!These parts are not running as one pipeline!!! No console output - only log (except when you run  **run.sh**)!!!

<br>All functions are wrapped with `@profile` decorator - it prints out into log file time and memory usage during 
<br>program execution<br>. The code of this utility is in **utils.py** ( in the head project directory).
<br>The project has only one **recipes_logging.log** file ( in the head project directory). The project has **run.sh**
<br>file, which you run in CLI's `bash run.sh htth://www.anylinkfromtestlinksfileindatafolder.com`. Consequently, it 
<br>gives the output in the terminal window a JSON-like text from the entered link page. In **recipes_logging.log** file will 
<br>be information about all functions ran: steps made, classification, probability, and metrics. 
<br>**requirements.txt** contains applied Python packages.
<br>**notebooks_and_drafts**  - just notebooks and drafts. The project run doesn't touch it.
<br>All constants (or almost all) in **config.py** 
## Methodological information
### **Scraping data from recipes website**

<br>Scraped data {'Recepie':[text],'INGREDIENTS': text} is in **data/recipes.pkl**
<br>The scraping part starts from **main_scraper.py**. It extracts all links from a page with all recipes using 
<br>_BeautifulSoup_ and asynchronous HTTP requests (_Grequests_). Collects all data into defaultdict, save into pickle file
<br>**data/new_recipe.pkl/**. Ten first URLs saves into **data/test_links.txt**. If you're going to test the model work 
<br>using **run.sh** file, then you can use these links.
<br> To continue run **preprocess.py**

### **Preprocessing, modeling, feature engineering**
Next stage is starting from run `main_preprocess(filename)` in **preprocess.py**.This function loads the data from 
<br>**data/recipes.pkl** file. Calls for `load_data_transform_to_set(filename)` to transform into data 
<br>set with column `paragraph` and `label`.Then calls for `utils.stratified_split_data(text, label, TEST_SIZE)`.
<br>After splitting, preprocess separately train/test sets with `preprocess_clean_data(train_dataset.as_numpy_iterator(), f'train')`.
<br>In  `preprocess_clean_data` function cleans the series of text and creates new columns with additional features. New 
<br>sets saves into two pkl files: **data/train_data_clean.pkl** and **data/test_data_clean.pkl**.
<br>It is the end of the first Data Science part. The next step is in **model_train.py**.  

### **Train, tune, evaluation**
<br>When you run **model_train.py**, at first, it will read **data/train_data_clean.pkl** 
<br>and **data/test_data_clean.pkl**. Then count `max len sent/sequence` and `vocabulary size`. Next will be call for 
<br>`preprocess.tfidf(texts, vocab_size) ` function to transform data into sequences. Then split data into nlp and meta 
<br>sets for test and train sets, call for
        `preprocess.get_model(tf_idf_train, X_meta_train, results, embedding_dimensions=EMBEDDING_DIM)`
<br> to create, build and train the MODEL. Next step is evaluation on the test sets with writing down results into log 
file, plot loss vs val_loss, save the model into `config.MODEL_NAME='data/my_model.h5'`. Here is the end of the modeling
<br>and preprocessing.

**_The last final stage:_** to run in terminal `bash run.sh htth://www.anylinkfromtestlinksfileindatafolder.com`. 
It call **main_task_run.py**. It accepts an URL as arguments . Then it  call for 
<br>**get_one.py**  to collect needed data from the page. We are
<br>assuming the url is valid and it redirects you to a page where a valid recipe is located. Next step - calls the 
<br>`utils.print_json(url_to_get_recipe, json_file)`, function which give you console output in JSON-like format.
<br>Next, apply transformation for the JSON file: from `list` ==> to `string` each element. Call the function 
<br>`preprocess.load_data_transform_to_set()` to transform from a defaultdict into data set. Preprocess text, engineer 
<br> new features, save to pikle file using `preprocess.preprocess_clean_data()`. Split into nlp and meta sets, call
<br> `eval_on_one_page(tfidf_one_page, X_meta_one_page, y_one_page, model, text)` from **model_train.py**. 
<br>All information about the metrics, model, table with predictions and probability values writes into log file.

<br>_**IMPORTANT:**_ If you want to test models in **/data** folder on test links, then you have to run **notebooks_and_drafts/list_dir.py**
<br> In **/data** folder can be several pretrained models. **notebooks_and_drafts/list_dir.py** runs with URLs from **data/test_links.txt**

## Data specific information

The data is collected from 'https://www.loveandlemons.com'
<br>Data is imbalanced 80/20. 
<br>Paragraph = all lines in ingredients labeled as 1, 
<br>Paragraph = each `\n` in Instructions labeled as 0.
<br>Vocabulary size = 2474 words/lemmas. It can be changed. It very depends on scrapping part -
<br>links to scrap each time goes in different order --> when you take out first 10 links  it will be 10 different links
<br>each scraper run --> the vocabulary every time will be different size
<br>Test split size = 0.2
<br>Data from 10 URLs in **data/test_links.txt** wasn't included into set for model train.