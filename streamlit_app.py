import os
import requests
import streamlit as st


st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

API_URL = os.getenv("API_URL", "http://localhost:8001")


def post_chat(messages, top_k=5, include_sources=True):
	url = f"{API_URL}/chat"
	resp = requests.post(url, json={
		"messages": messages,
		"top_k": top_k,
		"include_sources": include_sources,
	}, timeout=60)
	resp.raise_for_status()
	return resp.json()


st.title("RAG Chatbot (NEC + Wattmonk)")

if "history" not in st.session_state:
	st.session_state.history = []


with st.sidebar:
	st.markdown("### Settings")
	top_k = st.slider("Results per query", 2, 10, 5)
	include_sources = st.checkbox("Show sources", value=True)
	st.markdown("---")
	st.caption("API: " + API_URL)
	st.markdown("---")
	st.markdown("### Search")
	default_ns = st.selectbox("Namespace", ["nec", "wattmonk"]) 
	search_q = st.text_input("Find in knowledge base")
	if st.button("Search") and search_q:
		try:
			res = requests.post(f"{API_URL}/search", json={
				"query": search_q,
				"namespace": default_ns,
				"top_k": top_k,
			}, timeout=60)
			res.raise_for_status()
			hits = res.json()
			st.markdown("#### Results")
			for h in hits:
				st.write(f"- {h.get('source','')} · score: {h.get('score',0):.3f}")
				st.caption(h.get("snippet",""))
		except Exception as e:
			st.error(f"Search error: {e}")
    
user_input = st.chat_input("Ask a question...")

for msg in st.session_state.history:
	with st.chat_message(msg["role"]):
		st.markdown(msg["content"]) 

if user_input:
	st.session_state.history.append({"role": "user", "content": user_input})
	with st.chat_message("assistant"):
		with st.spinner("Thinking..."):
			try:
				res = post_chat(st.session_state.history, top_k=top_k, include_sources=include_sources)
				answer = res.get("answer", "")
				st.markdown(answer)
				if res.get("confidence") is not None:
					st.caption(f"Confidence: {res['confidence']:.3f}")
				if include_sources and res.get("sources"):
					st.markdown("### Sources")
					for s in res["sources"]:
						st.caption(f"- {s.get('source','')} (score: {s.get('score',0):.3f})")
				st.session_state.history.append({"role": "assistant", "content": answer})
			except Exception as e:
				st.error(f"Error: {e}")


