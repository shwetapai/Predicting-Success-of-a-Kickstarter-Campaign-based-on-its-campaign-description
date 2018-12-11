# Project5 :**Kickstarter Campaign Analyzer**

**Motivation**

Kickstarter is the world's largest funding platform for creative projects.It is useful for small enterpreners particularly those that may otherwise struggle to obtain capital for their projects.As of October 2018, only 36 % of the projects wwere successfully funded on kickstarter.com.

Although several studies have identified many antecedents that are associated with funding success,they mostly focus on basic project properties. The information related to project descriptions has not been explored thouroughly.Despite the difference between the funding environments, project descriptions are similar to traditional business plans in terms of both content and function.The project description is one of the most important information sources for backers to evaluate a project and make their funding decisions.Also, project descriptions are one of the things that creators can tweak easily before launching their campaign.

**Main Aim**

The main aim of my project was to identify the antecedents of a campaign's success based on its project description.

**Process**

I scrapped data on 7000 kickstrtes from the month Aug 2018.I decided to only focus on project whose pdeged goal was expressed in USD. My final dataset had 5800 kickstarter projects.

I then extracted information from project descriptions for each project and stored it in a dataframe.I then extracted 18 meta-features based on the project descriptions( 'num_sents', 'num_words', 'num_all_caps', 'percent_all_caps','num_exclms', 'percent_exclms', 'num_imp_words','percent_imp_words', 'avg_words_per_sent', 'num_paragraphs','avg_sents_per_paragraph', 'avg_words_per_paragraph','num_images', 'num_videos', 'num_youtubes','num_hyperlinks', 'num_bolded', 'percent_bolded').

I then performed sentiment analysis on the project description of each project and included 'sentiment' as one of the features.I then preprocessed the project descriptions ( by using tokenisers, vectorizer)  and used NMF to group the project description in 20 topics.I then included the 20 topics as features in the final dataset along with the 19 meta-features discussed above. My final dataset had 5082 rows and 39 columns.I decided to use all the features in my model as it gave the highest AUC.My target variable was binary ( 1:Funded, 0:Not Funded).


**Machine Learning Model**

After splitting the dataset into 'training' and 'test' sets, I tried various classifier models on the dataset. I finally selected random forest classifier and it was slightly better than other models in terms of its **AUC (0.77)**
 and the precision.I decided to focus on 'precision' as a perfomance metric as a false positve (Predicting that a project will be funded when it actually ends up not being funded) was a more serious error that the false negative. I didnot focus on false-negative as it's unlikely that any creator would abandon their painstaking efforts on their Kickstarter project after receiving an estimate from a single website.

I used the 'feature_importances_' feature of the model to find the most predictive features. Keeping the value of all other features constant, I  changed the value of the most predictive features by a small value to observe the change in the caipaign's probablity of success.Changing certain features reulted in a very modest improvement in the campaign's probablity of success.

**Results **

I utilized the above model in a flask app. The app takes a link to a kickstarter project as an input and reports the probablity of a project's success.I initially aslo included functions that recommend changes that could improve a project's probablity of success. I decided focus on the current probablity of success as the features changes led to a very modest improvement in the project's probablity of success.


**What Could I have done differently**

I think I should have focussed on individual words in a campaign description rather than on meta-fetures calculated from the campaign description.I could have started with using word2vect and cosine similarity to see which words in the description when replaced with certain words give the highest probablity of a campaign's success.That I feel would have been more productive and useful.An option which replaces some words with better high impact words is always useful.






