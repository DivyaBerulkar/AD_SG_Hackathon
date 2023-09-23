import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
# Sample academic documents (replace with your actual document database)
academic_documents = {
    'research_papers': 'This is the content of the first academic document. It discusses various research topics.',
    'academic_papers': 'The second document contains information about recent academic research findings in various subjects.',
    'text_books': 'Document three covers the best text books available for students and researchers.',
    # Add more documents here
}
# Step 1: Preprocess the academic documents
def preprocess_documents(documents):
    preprocessed_documents = {}
    for doc_id, document_text in documents.items():
        # Tokenize the document into sentences and words
        sentences = sent_tokenize(document_text)
        words = [word_tokenize(sentence.lower()) for sentence in sentences]

        # Remove stopwords and punctuation
        stop_words = set(stopwords.words('english'))
        words = [[word for word in sentence if word.isalnum() and word not in stop_words] for sentence in words]

        preprocessed_documents[doc_id] = {
            'sentences': sentences,
            'words': words,
        }

    return preprocessed_documents
# Step 2: Vectorize the academic documents
def vectorize_documents(preprocessed_documents):
    tfidf_vectorizers = {}
    tfidf_matrices = {}
    for doc_id, preprocessed_data in preprocessed_documents.items():
        sentences = preprocessed_data['sentences']

        # Use TF-IDF vectorization to convert text data into numerical vectors
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)

        tfidf_vectorizers[doc_id] = tfidf_vectorizer
        tfidf_matrices[doc_id] = tfidf_matrix

    return tfidf_vectorizers, tfidf_matrices
    from sklearn.metrics.pairwise import cosine_similarity
    # Step 3: Create a function to query academic documents
    def query_documents(query, preprocessed_documents, tfidf_vectorizers, tfidf_matrices):
        query_results = {}
        for doc_id, preprocessed_data in preprocessed_documents.items():
            sentences = preprocessed_data['sentences']

            # Preprocess the query
            query_words = word_tokenize(query.lower())
            query_words = [word for word in query_words if
                           word.isalnum() and word not in set(stopwords.words('english'))]
            # Transform the query into a TF-IDF vector
            query_vector = tfidf_vectorizers[doc_id].transform([' '.join(query_words)])

            # Calculate cosine similarity between the query vector and document vectors
            cosine_similarities = cosine_similarity(query_vector, tfidf_matrices[doc_id])

            # Rank sentences by similarity and return top results
            ranked_sentences = [(cosine_similarities[0][i], sentence) for i, sentence in enumerate(sentences)]
            ranked_sentences.sort(reverse=True)
            query_results[doc_id] = ranked_sentences

        return query_results
        import nltk
        nltk.download('punkt')
        nltk.download('stopwords')
        # Example usage:
        # Assuming you have already loaded academic documents into the 'academic_documents' dictionary
        preprocessed_docs = preprocess_documents(academic_documents)
        tfidf_vecs, tfidf_mats = vectorize_documents(preprocessed_docs)

        # Sample query
        user_query = 'research topics in mathematics'

        # Query the academic documents
        query_results = query_documents(user_query, preprocessed_docs, tfidf_vecs, tfidf_mats)

        # Display search results for each document
        for doc_id, results in query_results.items():
            print(f"Results for Document: {doc_id}")
            num_results_to_display = 3  # Display the top 3 results for each document
            for i in range(min(num_results_to_display, len(results))):
                print(f"Rank {i + 1}: Similarity Score = {results[i][0]:.2f}")
                print(results[i][1])
        print()