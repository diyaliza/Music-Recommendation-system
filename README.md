# Music-Recommendation-system
A music recommendation system built on Tensorflow recommendors using two tower archietcture

## DAC
<img width="401" alt="image" src="https://github.com/diyaliza/Music-Recommendation-system/assets/120042912/8c1682f8-ff68-4de0-9fca-77766b4e256b">

## Focus Questions
1. What are the recommended songs for a particular user who has already listened to a particular song?
2. What is the most popular genre of music?
3. Is there a correlation between the genre of the music and the popularity of the music ?
4. What are the features of music that makes a music more popular?
5. How do music preferences change over time of the day, week or month ?

## Modeling
 
We use 2 tower architecture for the modeling. In the Modeling step of the CRISP-DM framework, we define and build the machine learning models based on the understanding gained from the data and business objectives. In the music recommendation system project, this step involves creating three distinct models: UserModel, SongModel, and CandidatesModel. Here's an overview of the modeling step:

### UserModel
The UserModel is a subclass of tf.keras.Model. In its __init__ method, it defines two sequential models (user_query_model and song_query_model) within the model. 
#### user_query_model
This part of the model processes user-related information. 
InputLayer: Takes an input tensor of shape (1,) representing user IDs as strings. 
StringLookup: Converts user IDs into numerical indices based on a vocabulary (unique_user_ids).
Embedding: Maps the numerical indices to dense vectors of fixed size (output_dim=32 in this case).
Flatten: Flattens the output to a one-dimensional tensor.
#### song_query_model
Similar to the user_query_model, this part processes song-related information.
Takes an input tensor of shape (1,) representing song IDs as strings. 
Converts song IDs into numerical indices based on a vocabulary (unique_song_ids). 
Maps the numerical indices to dense vectors using embedding. 
Flattens the output.
#### call Method
The call method defines how inputs should be processed through the model. It takes a dictionary of inputs (inputs), which is assumed to contain keys 'user_id' and 'song_id'. It concatenates the outputs of user_query_model and song_query_model along the second axis.

In summary, the UserModel processes user and song information separately, converting string IDs into numerical representations using embedding layers. The outputs are then concatenated, forming a combined representation that can be used in subsequent layers of the recommendation system. This modular design allows for flexibility and reuse in building complex models.
### SongModel
The SongModel is a subclass of tf.keras.Model. In its __init__ method, it defines four sequential models (song_candidate_model, genre_candidate_model, title_text_embedding, and total_views) within the model.

#### song_candidate_model
Processes song-related information based on the song ID. InputLayer: Takes an input tensor of shape (1,) representing song IDs as strings.
StringLookup: Converts song IDs into numerical indices based on a vocabulary (unique_song_ids).
Embedding: Maps the numerical indices to dense vectors of fixed size (output_dim=32).
Flatten: Flattens the output to a one-dimensional tensor.

#### genre_candidate_model
Similar to song_candidate_model, this part processes genre-related information.
Takes an input tensor of shape (1,) representing genre IDs as strings.
Converts genre IDs into numerical indices based on a vocabulary (unique_genre_ids).
Maps the numerical indices to dense vectors using embedding.
Flattens the output.

#### title_text_embedding
Processes the title text of songs.
Takes an input tensor of shape (1,) representing title texts as strings.
Converts title texts into numerical indices based on a vocabulary (unique_title_song_ids).
Maps the numerical indices to dense vectors using embedding, allowing for variable-length sequences (mask_zero=True).
Flattens the output.

#### total_views
Processes the total views information.
Takes an input tensor of shape (1,) representing total views as integers.
Discretizes the total views using a specified set of buckets.
Converts discretized views into numerical indices using embedding.
Flattens the output.

#### call Method
The call method defines how inputs should be processed through the model. It takes a dictionary of inputs (titles), which is assumed to contain keys 'song_id', 'song_title', 'total_views', and 'genre1'. It concatenates the outputs of the various sub-models along the second axis.
In summary, the SongModel processes different aspects of song-related information, including song ID, genre, title text, and total views. The outputs of these sub-models are concatenated to form a combined representation that captures various features of a song. This modular design allows for flexibility in handling diverse information related to songs in the recommendation system.

### CandidatesModel
The CandidatesModel is a subclass of tfrs.models.Model, indicating that it's designed for recommendation tasks using TensorFlow Recommenders (TFRS). The model consists of two main components: a query_model and a candidate_model, each defined as sequential models.

#### query_model
It utilizes the previously defined UserModel to process user-related information. It adds a dense layer with 32 units after the UserModel.

#### candidate_model
It utilizes the previously defined SongModel to process song-related information. It adds a dense layer with 32 units after the SongModel.

#### task
The model includes a retrieval task (tfrs.tasks.Retrieval) with a specified metric (tfrs.metrics.FactorizedTopK). The metric involves providing candidates for the top-K recommendations. In this case, it uses the candidates obtained from batching and mapping the candidate_model over the songs dataset.

#### compute_loss Method
This method defines how to compute the loss during training. It takes a dictionary of input features (features) and a boolean flag (training). It obtains query embeddings using the query_model for user-related features. It obtains candidate embeddings using the candidate_model for song-related features. The loss is computed using the retrieval task (self.task) by comparing query embeddings to candidate embeddings.

In summary, the CandidatesModel integrates user and song information through the query_model and candidate_model, respectively. It defines a retrieval task with a metric for evaluating recommendations. The compute_loss method specifies how to calculate the loss during training, facilitating the optimization of the model for effective recommendations. This model is designed to learn the embeddings that best match user preferences with suitable song candidates.
