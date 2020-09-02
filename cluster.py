from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score


def kmeans_key_word(words, vectors, max_k=5, random_seed=128, top_n=2):
    fit_k = 0
    max_score = 0
    unique_words = len(set(words))
    if max_k > 2:
        kmeans_res_set = {}
        for k in range(2, max_k + 1):
            if k >= unique_words:
                continue
            kmeans = KMeans(n_clusters=k, random_state=random_seed)
            kmeans.fit(vectors)

            y_predict = kmeans.labels_
            score = silhouette_score(vectors, y_predict)
            kmeans_res_set[k] = kmeans
            if score > max_score:
                max_score = score
                fit_k = k
        kmeans = kmeans_res_set[fit_k]
    else:
        kmeans = KMeans(n_clusters=2, random_state=random_seed)
        kmeans.fit(vectors)

    centers = kmeans.cluster_centers_

    res = {}

    for index, cluster_id in enumerate(kmeans.labels_):
        score = cosine_similarity([vectors[index], centers[cluster_id]])[0][1]

        if cluster_id in res:
            if words[index] in res[cluster_id]:
                res[cluster_id][words[index]] = max(res[cluster_id][words[index]], score)
            else:
                res[cluster_id][words[index]] = score
        else:
            res[cluster_id] = {}
            res[cluster_id][words[index]] = score
    keywords = []
    for i in res:
        after = dict(sorted(res[i].items(), key=lambda e: e[1], reverse=True))
        key_value = list(after.keys())
        keywords.append(key_value[:top_n])
    return keywords


if __name__ == '__main__':
    from tencent_word_embedding import WordEmbedding

    _tencent_word_embedding = WordEmbedding("conf/Tencent_AILab_ChineseEmbedding/Tencent_AILab_ChineseEmbedding.bin")
    _tencent_embeddor = _tencent_word_embedding.word_vector

    _words = ['福利', '腾讯', '汽车', '会员', '中石油', '中石化', '充值', '九折卡', '驾乘', '意外险']
    _vectors = []

    for token in _words:
        _vectors.append(_tencent_embeddor(token))
    print('LOADED VECTORS')
    _keywords = kmeans_key_word(_words, vectors=_vectors, max_k=5, random_seed=128, top_n=2)
    _keywords = [y for x in _keywords for y in x]

    print(_keywords)
