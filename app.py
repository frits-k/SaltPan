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

st.set_page_config(
    page_title="Convert Voice Memos to Text | Learn LangChain",
    page_icon="🔊"
)

def transcribe_chunk(chunk, chunk_number):
	chunk_name = f"temp_chunk_{chunk_number}.wav"
	chunk.export(chunk_name, format="wav")

	with open(chunk_name, "rb") as audio_file:
		transcript = openai.audio.transcriptions.create(model="whisper-1", file=audio_file)

	os.remove(chunk_name)
	return transcript.text

def transcribe_long_audio(file_path, file_name, chunk_length_ms=120000):  # 2 minutes chunks
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

st.header('🔊 Convert Voice Memos to Text')

st.subheader('Learn LangChain | Demo Project #5')

st.success("This is a demo project related to the [Learn LangChain](https://learnlangchain.org/) mini-course.")

st.write('''
This demo project takes inspiration from real life. I was reading a nutrition book and taking some
audio notes/voice memos to keep track of the most useful information. Once finished the book, I
thought that it would be useful to put the information together in an organic document, and that's
really the kind of task you can automate with LangChain and LLM.

In this tool, we build a simplified version of a custom LangChain document loader, to transcribe the
audio using the OpenAI Whisper model and return it in the standardized LangChain format. This would
not have been a required step, but in case we want to store the audios, split them or create more
elaborated flows, it's always nice to stick with the LangChain default document format.

The tool can transcribe the voice memos as they are, or you can provide a custom prompt to adjust
the tone, translate into another language, fix the grammar or the form, or - like in my case - organize
the transcripts into book chapters. Sky is the limit!''')

st.info("You need your own keys to run commercial LLM models.\
    The form will process your keys safely and never store them anywhere.", icon="🔒")

openai.api_key = st.text_input("OpenAI Api Key", help="You need an account on OpenAI to generate a key: https://openai.com/blog/openai-api")

voice_memos = st.file_uploader("Upload your voice memos", type=["m4a", "mp3"], accept_multiple_files=True)

post_processing = st.checkbox('Post-process your text transcript with a custom prompt')

with st.form("audio_text"):	

	if post_processing:

		model = st.selectbox(
			'Select a model',
			('gpt-3.5-turbo','gpt-4'),
			help="Make sure your account is eligible for GPT4 before using it"
		)

		custom_prompt = st.text_area("Custom prompt")

		st.write('''
		To further process your transcript effectively, the prompt should start with:
		"Given the following transcript...". Here are a few examples:
		- Given the following transcript, please change the tone of the voice and make it very formal.
		- Given the following transcript, please translate it to *
		- Given the following transcript, please summarize it in * words making sure the core concepts are included
		''')

	execute = st.form_submit_button("🖊️ Process Voice Memos")

	if execute:

		with st.spinner('Converting your voice memos...'):

			if voice_memos is not None:

				for voice_memo in voice_memos:

					file_name, file_extension = os.path.splitext(voice_memo.name)

					with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temporary_file:
						temporary_file.write(voice_memo.read())

					audio_doc = transcribe_long_audio(temporary_file.name, file_name)

					if post_processing:

						llm = ChatOpenAI(openai_api_key=openai_key, temperature=0, model_name=model)

						prompt = ChatPromptTemplate.from_template('''
						{prompt}
						{transcript}
						''')

						chain = LLMChain(llm=llm, prompt=prompt)

						response = chain.run({
							'prompt': custom_prompt,
							'transcript': audio_doc.page_content
						})
						
						st.write(response)

					else:

						st.write(audio_doc.page_content)

					# clean-up the temporary file
					os.remove(temporary_file.name)

with st.expander("Exercise Tips"):
	st.write('''
	This demo is probably the most interesting one to expand and improve:
	- Browse [the code on GitHub](https://github.com/francescocarlucci/wordpress-code-assistant/blob/main/app.py) and make sure you understand it.
	- Fork the repository to customize the code.
	- Try to rewrite the document loader as a Class, and give it the same structure as others LangCHain loaders (TextLoader, CSVLoader).
	- Get creative and produce more post processing flows, maybe enriching the UI as well. 
	''')

st.divider()

st.write('A project by [Francesco Carlucci](https://francescocarlucci.com) - \
Need AI training / consulting? [Get in touch](mailto:info@francescocarlucci.com)')