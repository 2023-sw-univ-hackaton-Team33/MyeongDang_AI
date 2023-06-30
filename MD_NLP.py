from flask import Flask, request, jsonify
from konlpy.tag import Komoran
from summa import keywords, summarizer
import re
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)

@app.route('/summarize_text', methods=['POST'])
def summarize_text():
    data = request.get_json()
    companyName = data.get('companyName')
    news_url = f"https://news.google.com/search?q={companyName}&hl=ko&gl=KR&ceid=KR%3Ako"

    response = requests.get(news_url)
    soup_news = BeautifulSoup(response.text, 'html.parser')
    articles = soup_news.select('h3 > a')

    news_titles = []
    news_urls = []

    for article in articles[:3]:
        title = article.get_text()
        news_titles.append(title)
        link = "https://news.google.com" + article['href']
        news_urls.append(link)
   # print(news_titles)
   # print(news_urls)

    news_summary = []

    for i in range(3): # 따온 뉴스의 url에 들어가 모든 값 크롤링 해옴
        url = news_urls[i]  # 키가 "article"인 값을 추출
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers={'User-agent' : 'Mozilla/5.0'})
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "lxml")
        text = soup.get_text()

        text = re.sub(r'\s+', ' ', text) #여기에 데이터베이스에서 가져온 뉴스 기사 본문
        #print(text)

        max_length = 200 #최대 글자수
        komoran = Komoran()
        # 키워드 추출
        text_keywords = keywords.keywords(text).split('\n')
        # 문장별로 점수 계산
        sentence_scores = {}
        sentences = text.split('.')
        for sentence in sentences:
            score = 0
            for keyword in text_keywords:
                if keyword in sentence:
                    score += 1
            sentence_scores[sentence] = score
        # 문장 중요도에 따라 정렬
        sorted_sentences = sorted(sentence_scores.items(), key=lambda item: item[1], reverse=True)
        # 상위 문장 선택하여 요약
        summary_sentences = [sentence for sentence, score in sorted_sentences[7:9]]  # 상위 3개 문장 선택
        summary = " ".join(summary_sentences)
        if len(summary) > max_length:
            summary = summary[:max_length] + "…"
        news_summary.append(summary)
    print(news_summary)

    news = []
    for i in range(3):
        news.append({
            "title" : news_titles[i],
            "url" : news_urls[i],
            "summary" : news_summary[i]
        })
    return jsonify({'news': news}, ensure_ascii=False)

@app.route('/extract_keywords', methods=['POST'])
def extract_keywords():
    top_k = 5 # 키워드 개수
    data = request.get_json()
    news_url = data.get('url')  # 키가 "article"인 값을 추출
    # 문장에서 한글을 제외한 문자 제거
    #text = re.sub(r"[^ㄱ-ㅎㅏ-ㅣ가-힣\s]", "", text)
    
    # 형태소 분석기를 활용한 토큰화
    tokenizer = Okt()
    tokens = tokenizer.pos(text, stem=True)
    
    # 명사, 형용사, 부사만 추출하여 키워드로 사용
    keywords = [word for word, pos in tokens if pos in ['Noun', 'Adjective']]
    
    # TF-IDF 벡터화
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform([' '.join(keywords)])
    
    # TF-IDF 값이 높은 상위 키워드 추출
    feature_names = tfidf.get_feature_names_out()
    stopwords = ['있다', '이다']
    feature_names = [keyword for keyword in feature_names if keyword not in stopwords]
    top_keywords = []
    if len(feature_names) > 0:
        sorted_indices = tfidf_matrix.toarray().argsort(axis=1)[:, ::-1]
        top_indices = sorted_indices[:, :top_k].flatten()
        top_keywords = [feature_names[idx] for idx in top_indices]
    return jsonify({'keywowrds' : top_keywords })

if __name__ == '__main__':
    app.run()
