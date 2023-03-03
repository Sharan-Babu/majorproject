import streamlit as st
import streamlit.components.v1 as components
import pickle
from newscatcherapi import NewsCatcherApiClient
from sentence_transformers import SentenceTransformer, util



st.title('Gemini 👥')
st.caption('Eclectic News Search Engine')
API_KEY = '3h7H4_yMu2KchsDvXCt2x5TZ4m002_G8myOud-tPjl8'
newscatcherapi = NewsCatcherApiClient(x_api_key=API_KEY)
  
threshold = 0.8

@st.cache_resource
def load_model():
	return SentenceTransformer('msmarco-distilbert-base-tas-b')


search_term = st.text_input("Enter query","russia vs ukraine")

if st.button("Search"):
	embedder = load_model()
	news_articles = newscatcherapi.get_search(q=search_term, lang='en', page_size=5)
	news_articles = news_articles['articles']

	anchor_vertices_list = []
	children_news_info = []
	sentences_list = []

	for x in news_articles:
		if len(anchor_vertices_list) == 0:
			anchor_vertices_list.append(x)
			children_news_info.append([])
			sentences_list.append(x['summary'])
		else:
			main_article_summary = x['summary']
			main_article_embedding = embedder.encode(main_article_summary)
			remaining_articles_embeddings = embedder.encode(sentences_list)
			cos_sim = util.cos_sim(main_article_embedding, remaining_articles_embeddings)[0]
			closest_to = -1
			sim = 0
			index = 0
			for value in cos_sim:
				if float(value) > threshold and float(value) > sim:
					closest_to = index
					sim = float(value)
				index += 1
			
			if closest_to == -1:
				anchor_vertices_list.append(x)
				children_news_info.append([])
				sentences_list.append(x['summary'])
			else:
				children_news_info[closest_to].append(x)
							



	#main_article_embedding = embedder.encode(main_article_summary)
	#remaining_articles_embeddings = embedder.encode(remaining_articles_array)
	#cos_sim = util.cos_sim(main_article_embedding, remaining_articles_embeddings)
	#cos_sim[0] -- len(99)  # illustrates similarty between first news article and remaining 99 news articles in our dataset


	# with open('anchor_vertices_list','rb') as file:
	#     anchor_vertices_list = pickle.load(file)

	# with open('children_news_info','rb') as file:
	#     children_news_info = pickle.load(file)    

	#st.write(anchor_vertices_list)
	#st.write(children_news_info)



	# Create 2 columns view
	st.markdown("# Anchor News")

	for i,article in enumerate(anchor_vertices_list):
		link = article['link']
		media = article['media']
		rank = article['rank']
		summary = article['summary']
		title = article['title']
		children = children_news_info[i]

		st.subheader(f"{i+1}. {title}")
		st.write(f"News Link: {link}")
		st.image(media,caption="News Thumbnail")
		st.markdown("#### Summary")
		st.write(summary)
		
		if len(children)>0:
			for y in children:
				with st.expander(y['title']):
					st.write(f"News Link: {y['link']}")
					#st.image(y['media'],caption="News Thumbnail")
					st.markdown("#### Summary")
					st.write(y['summary'])

		st.write("")
		st.write("")	
		st.markdown("<hr>",unsafe_allow_html=True)
