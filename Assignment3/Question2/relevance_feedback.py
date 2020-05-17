import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from evaluation import evaluate_retrieval

def relevance_feedback(vec_docs, vec_queries, sim, gt, n=10):
    """
    relevance feedback
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """
    
    #rf_sim_matrix = cosine_similarity(vec_docs , vec_queries)
    #np.argsort(-rf_sim_matrix)[:n]
    #print("vec_docs ", vec_docs.shape)
    #print("vec_queries ", vec_queries)
    #print("vec ", vec_queries[1].shape)
    #print("sim ", sim)
    
    #print("sim_matrix ", sim.shape)
    #print("sim_matrix ", sim)
    
    vec_docs = vec_docs.toarray()
    vec_queries = vec_queries.toarray()
    

    
    for c in range(3):
        print(c)
        for i in range(30):
            ranked_documents = np.argsort(-sim[:, i])[:n]

            rel = np.zeros(vec_docs[0].shape)
            #print("Rel shape ", rel.shape)
            not_rel = np.zeros(vec_docs[0].shape)
            for document in ranked_documents:
                flag=0
                for gt_docs in gt:
                    if(gt_docs[0] == i+1 and gt_docs[1] == document+1):
                        rel = rel + vec_docs[document]
                        flag=1
                        break
                if(flag==0):
                    not_rel = not_rel + vec_docs[document]
                    
            vec_queries[i] = vec_queries[i] + 0.5*rel - 0.9*not_rel
        sim = cosine_similarity(vec_docs, vec_queries)
    
    rf_sim = cosine_similarity(vec_docs, vec_queries)
    
    #rf_sim = sim # change
    return rf_sim


def relevance_feedback_exp(vec_docs, vec_queries, sim, tfidf_model, gt, n=10):
    """
    relevance feedback with expanded queries
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        tfidf_model: TfidfVectorizer,
            tf_idf pretrained model
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """
    
    vec_docs = vec_docs.toarray()
    vec_queries = vec_queries.toarray()
    #print(vec_queries)
    #print("Bansal ", vec_queries[28,7523])
    
    for c in range(3):
        print(c)
        for i in range(30):
            ranked_documents = np.argsort(-sim[:, i])[:n]
            rel = np.zeros(vec_docs[0].shape)
            not_rel = np.zeros(vec_docs[0].shape)
            rel_docs = []
            for document in ranked_documents:
                flag=0
                for gt_docs in gt:
                    if(gt_docs[0] == i+1 and gt_docs[1] == document+1):
                        rel = rel + vec_docs[document]
                        rel_docs.append(document)
                        flag=1
                        break
                if(flag==0):
                    not_rel = not_rel + vec_docs[document]
            
            #print(rel)
            vec_queries[i] = vec_queries[i] + 0.5*rel - 0.9*not_rel
            
            #print("Himanshu ", vec_queries[28,7523])
            
            sorted_vec_docs_index = []
            sorted_vec_docs_value = []
            sorted_vec_docs_index2 = []
            
            #print(rel_docs)
            for j in rel_docs:
                t = np.argsort(-vec_docs[j,:])[:n]
                sorted_vec_docs_index.append(t)
                for k in t:
                    sorted_vec_docs_value.append(vec_docs[j,k])
                    sorted_vec_docs_index2.append(k)
                    
            #print(sorted_vec_docs_index2)
            #print(sorted_vec_docs_value)
            
            max_index_list = []
            max_value_list = []
            for r in range (10):
                max_value = 0
                max_index = -1
                for index in range(len(sorted_vec_docs_index2)):
                    if(sorted_vec_docs_value[index] > max_value and sorted_vec_docs_index2[index] not in max_index_list):
                        max_value = sorted_vec_docs_value[index]
                        max_index = sorted_vec_docs_index2[index]
                max_index_list.append(max_index)
                max_value_list.append(max_value)
                
                
            #print(max_index_list)
            #print(max_value_list)
            for w in range(10):
                #print(max_value_list[w])
                #print("one " ,vec_queries[i,max_index_list[w]])
                #print("i ", i)
                #print("max ", max_index_list[w])
                vec_queries[i,max_index_list[w]] = vec_queries[i,max_index_list[w]] + max_value_list[w] 
                #print("two ", vec_queries[i,max_index_list[w]])
                
        sim = cosine_similarity(vec_docs, vec_queries)
    
    rf_sim = cosine_similarity(vec_docs, vec_queries)

    #print(vec_queries[28,7523])
    #rf_sim = sim  # change
    return rf_sim