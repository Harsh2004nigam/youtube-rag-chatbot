import os
from dotenv import load_dotenv
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_huggingface.chat_models import ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import re


# Load environment variables from .env (only local)
load_dotenv()

# Ensure your token is available
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")

# --- Streamlit UI setup ---
st.set_page_config(page_title="🎥 YouTube Chatbot", page_icon="🤖")
st.title("🎥 YouTube Chatbot using LangChain (RAG-based)")
st.markdown("Ask questions from any **YouTube video** that has English subtitles!")

# --- Extract YouTube Video ID ---
def extract_video_id(url):
    """
    Supports full URLs or direct video IDs.
    """
    if "youtube.com" in url or "youtu.be" in url:
        pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
        match = re.search(pattern, url)
        return match.group(1) if match else None
    return url.strip()

# --- User input ---
yt_input = st.text_input("Enter YouTube URL or Video ID:")

if yt_input:
    video_id = extract_video_id(yt_input)
    if not video_id:
        st.error("⚠️ Invalid YouTube URL or Video ID.")
    else:
        with st.spinner("Fetching transcript..."):
            try:
                # ✅ Correct method
                transcript_data = YouTubeTranscriptApi().fetch(video_id, languages=['en'])
                transcript = " ".join(snippet.text for snippet in transcript_data)
                st.success("✅ Transcript successfully fetched!")

                # --- Split text into chunks ---
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.create_documents([transcript])

                # --- Embeddings + Vector Store ---
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                vector_store = FAISS.from_documents(chunks, embeddings)
                retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

                # --- LLM Setup ---
                llm = HuggingFaceEndpoint(
                    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
                    task="text-generation"
                )
                model = ChatHuggingFace(llm=llm)

                # --- Prompt Template ---
                prompt = PromptTemplate(
                    template="""
                    You are a helpful assistant.
                    Use ONLY the provided transcript context to answer the question.
                    If the context is insufficient, say you don’t know.
                    Give the answer in bullet points.

                    Context:
                    {context}

                    Question: {question}
                    """,
                    input_variables=['context', 'question']
                )

                # --- Query input ---
                query = st.text_input("💬 Ask a question about this video:")

                if query:
                    with st.spinner("Generating answer..."):
                        retrieved_docs = retriever.invoke(query)
                        context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

                        def format_docs(retrieved_docs):
                            return "\n\n".join(doc.page_content for doc in retrieved_docs)

                        parallel_chain = RunnableParallel({
                            'context': retriever | RunnableLambda(format_docs),
                            'question': RunnablePassthrough()
                        })

                        parser = StrOutputParser()
                        main_chain = parallel_chain | prompt | model | parser

                        answer = main_chain.invoke(query)
                        st.markdown("### 🧠 Answer:")
                        st.write(answer)

            except TranscriptsDisabled:
                st.error("❌ This video has **no subtitles** available.")
            except NoTranscriptFound:
                st.error("❌ Could not find English transcript for this video.")
            except Exception as e:
                st.error(f"❌ Error fetching transcript: {e}")
