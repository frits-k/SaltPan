import os
import openai
import tempfile
import streamlit as st
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.docstore.document import Document
from pydub import AudioSegment
from pydub.utils import make_chunks

def transcribe_chunk(chunk, chunk_number):
	chunk_name = f"temp_chunk_{chunk_number}.mp3"
	chunk.export(chunk_name, format="mp3")

	with open(chunk_name, "rb") as audio_file:
		transcript = openai.audio.transcriptions.create(model="whisper-1", file=audio_file)

	os.remove(chunk_name)
	return transcript.text

def transcribe_long_audio(file_path, file_name, chunk_length_ms=1200000):  # 2 minutes chunks
	audio = AudioSegment.from_mp3(file_path)
	chunks = make_chunks(audio, chunk_length_ms)

	full_transcript = ""
	for i, chunk in enumerate(chunks):
		print(f"Transcribing chunk {i + 1} of {len(chunks)}...")
		chunk_transcript = transcribe_chunk(chunk, i)
		full_transcript += chunk_transcript + " "

	return Document(
		page_content = full_transcript.strip(),
		metadata = {'file_name': file_name}
	)

st.info("You need your own keys to run commercial LLM models.\
    The form will process your keys safely and never store them anywhere.", icon="🔒")

openai.api_key = st.text_input("OpenAI Api Key", help="You need an account on OpenAI to generate a key: https://openai.com/blog/openai-api")

voice_memos = st.file_uploader("Upload your voice memos", type=["m4a", "mp3"], accept_multiple_files=True)

with st.form("audio_text"):
	execute = st.form_submit_button("🖊️ Process Voice Memos")

	if execute:
		with st.spinner('Converting your voice memos...'):
			if voice_memos is not None:
				for voice_memo in voice_memos:
					file_name, file_extension = os.path.splitext(voice_memo.name)

					with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temporary_file:
						temporary_file.write(voice_memo.read())

					audio_doc = transcribe_long_audio(temporary_file.name, file_name)

					with open("prompt.txt", "r") as file:
						custom_prompt = file.read()
					llm = ChatOpenAI(openai_api_key=openai.api_key, temperature=0, model_name="gpt-3.5-turbo")

					prompt = ChatPromptTemplate.from_template('''
					{prompt}
					
					Here is the transcript:
					{transcript}
					
					Please use the above transcript to generate the Graphviz code as specified.
					''')

					chain = LLMChain(llm=llm, prompt=prompt)

					response = chain.run({
						'prompt': custom_prompt,
						'transcript': audio_doc.page_content
					})

					st.write(response)

					# Clean up the temporary file after each processing loop
					os.remove(temporary_file.name)
			else:
				st.write("No audio file uploaded.")

st.divider()

st.write('A project by [Francesco Carlucci](https://francescocarlucci.com) - \
Need AI training / consulting? [Get in touch](mailto:info@francescocarlucci.com)')