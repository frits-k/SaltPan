import ffmpeg
import os
import openai
import tempfile
import streamlit as st
from langchain.chains import LLMChain  # Updated import for LLMChain
from langchain_community.chat_models import ChatOpenAI  # Updated import for ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate  # Updated import for ChatPromptTemplate
from langchain.docstore.document import Document
from pydub import AudioSegment
from pydub.utils import make_chunks

st.set_page_config(
    page_title="SaltPan",  # Title of the app shown in the browser tab
    page_icon="🌊",           # Optional: Icon shown in the browser tab
    layout="wide",        # Optional: Layout of the app ('centered' or 'wide')
    initial_sidebar_state="auto"  # Optional: Sidebar state ('auto', 'expanded', 'collapsed')
)

# Custom HTML/CSS for the banner
custom_html = """
<div class="banner">
    <img src="https://saltassociation.co.uk/wp-content/uploads/salt-crystal.jpg" alt="Banner Image">
</div>
<style>
    .banner {
        width: 100%;
        height: 200px;
        overflow: hidden;
    }
    .banner img {
        width: 100%;
        object-fit: cover;
    }
</style>
"""
# Display the custom HTML
st.components.v1.html(custom_html)

def transcribe_chunk(chunk, chunk_number):
	chunk_name = f"temp_chunk_{chunk_number}.mp3"
	chunk.export(chunk_name, format="mp3")

	with open(chunk_name, "rb") as audio_file:
		transcript = openai.audio.transcriptions.create(model="whisper-1", file=audio_file)

	os.remove(chunk_name)
	return transcript.text

def check_and_convert_to_128kbps(file_path):
	"""
	Checks if the audio file is an MP3 with 128 kbps or a WAV file.
	Converts to a 128 kbps MP3 if necessary.
	Returns the path to the (possibly converted) MP3 file and whether it was converted.
	"""
	file_extension = os.path.splitext(file_path)[1].lower()

	if file_extension == ".mp3":
		# Handle MP3 file
		probe = ffmpeg.probe(file_path)
		bitrate = int(probe['format']['bit_rate'])  # Bitrate is in bits per second

		if bitrate != 128000:  # 128 kbps is 128,000 bits per second
			print(f"Converting {file_path} to 128 kbps...")
			temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")  # Create a temporary file
			temp_file.close()
			output_path = temp_file.name

			# Convert the file to 128 kbps using ffmpeg
			ffmpeg.input(file_path).output(output_path, audio_bitrate="128k").run(overwrite_output=True)
			return output_path, True  # Return the converted file path and a flag
		else:
			print(f"{file_path} is already 128 kbps.")
			return file_path, False  # Return the original file path and a flag

	elif file_extension == ".wav":
		# Handle WAV file
		print(f"Converting {file_path} from WAV to 128 kbps MP3...")
		temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")  # Create a temporary file
		temp_file.close()
		output_path = temp_file.name

		# Convert the WAV file to 128 kbps MP3 using ffmpeg
		ffmpeg.input(file_path).output(output_path, audio_bitrate="128k").run(overwrite_output=True)
		return output_path, True  # Return the converted file path and a flag

	else:
		raise ValueError(f"Unsupported file format: {file_extension}. Only MP3 and WAV files are supported.")


def transcribe_long_audio(file_path, file_name, chunk_length_ms=1200000):  # 2 minutes chunks
	"""
	Splits the MP3 file into smaller chunks and transcribes each chunk.
	Ensures the MP3 is 128 kbps before processing.
	"""
	# Ensure the MP3 file is 128 kbps
	file_path, was_converted = check_and_convert_to_128kbps(file_path)

	# Load the audio file and split into chunks
	audio = AudioSegment.from_mp3(file_path)
	chunks = make_chunks(audio, chunk_length_ms)

	full_transcript = ""
	for i, chunk in enumerate(chunks):
		print(f"Transcribing chunk {i + 1} of {len(chunks)}...")
		chunk_transcript = transcribe_chunk(chunk, i)
		full_transcript += chunk_transcript + " "

	# Return the converted flag for cleanup
	return Document(
		page_content=full_transcript.strip(),
		metadata={'file_name': file_name}
	), was_converted, file_path  # Include file_path for cleanup

#st.info("You need your own keys to run commercial LLM models. The form will process your keys safely and never store them anywhere.", icon="🔒")

openai.api_key = st.text_input("Enter OpenAI Api Key", help="You need an account on OpenAI to generate a key: https://openai.com/blog/openai-api")

#voice_memo = st.file_uploader("Upload your voice recording", type=["mp3"])
voice_memo = st.audio_input("Pour in your thoughts")

with st.form("audio_text"):
	execute = st.form_submit_button("💠️Crystallize thoughts")

	if execute:
		with st.spinner("Crystallizing"):
			if voice_memo is not None:
				# Define a temporary file for storing the WAV data
				with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temporary_file:
					# Save the WAV data to a temporary file
					temporary_file.write(voice_memo.read())
					temp_file_path = temporary_file.name

				# Convert the WAV to a 128 kbps MP3 if necessary
				converted_file_path, was_converted = check_and_convert_to_128kbps(temp_file_path)

				# Extract the file name without extension
				file_name = os.path.splitext(os.path.basename(temp_file_path))[0]
				print(file_name)

				# Transcribe the audio and get conversion details
				audio_doc, was_converted, final_file_path = transcribe_long_audio(converted_file_path, file_name)
				# Step 1: Sanitize the transcript
				with open("prompt_sanitize.txt", "r") as file:
					sanitize_prompt = file.read()
				llm_sanitize = ChatOpenAI(openai_api_key=openai.api_key, temperature=0, model_name="gpt-3.5-turbo")

				sanitize_template = ChatPromptTemplate.from_template('''
				{prompt}

				Here is the transcript to sanitize:
				{transcript}
				''')

				sanitize_chain = LLMChain(llm=llm_sanitize, prompt=sanitize_template)

				# Run the sanitization step
				sanitized_transcript = sanitize_chain.run({
					'prompt': sanitize_prompt,
					'transcript': audio_doc.page_content
				})

				# Step 2: Generate Graphviz code from the sanitized transcript
				with open("prompt_graph.txt", "r") as file:
					graph_prompt = file.read()
				llm_graph = ChatOpenAI(openai_api_key=openai.api_key, temperature=0, model_name="gpt-3.5-turbo")

				graph_template = ChatPromptTemplate.from_template('''
				{prompt}

				<transcript>{sanitized_transcript}</transcript>

				Please use the above transcript to generate the Graphviz code as specified.
				''')

				graph_chain = LLMChain(llm=llm_graph, prompt=graph_template)

				response = graph_chain.run({
					'prompt': graph_prompt,
					'sanitized_transcript': sanitized_transcript
				})

				# Save response to session state and display it
				st.session_state["response"] = response
				#st.write(response)

				# Clean up temporary files
				#os.remove(temporary_file.name)
				#if was_converted:
				#	os.remove(temp_file_path)  # Remove the converted file if it exists
			else:
				st.write("No audio file uploaded.")

#st.divider()

def load_graph_from_script(script, api_key):
	try:
		# Attempt to execute the Graphviz script
		local_vars = {"graphviz": graphviz}  # Pass graphviz into the exec context
		exec(script, {}, local_vars)
		dot = local_vars["dot"]  # Retrieve the `dot` object
		return dot
	except Exception as e:
		# Handle invalid Graphviz script
		st.warning(f"Invalid Graphviz script detected: {e}. Attempting to correct it...")

		# Correct the script using OpenAI
		correction_prompt_template = """
		The following Graphviz code contains errors. Please correct it and return the fixed code. Do not include explanations, only the corrected code:

		{script}
		"""
		correction_prompt = ChatPromptTemplate.from_template(correction_prompt_template)
		llm = ChatOpenAI(openai_api_key=api_key, temperature=0, model_name="gpt-3.5-turbo")
		chain = LLMChain(llm=llm, prompt=correction_prompt)

		# Generate corrected script
		corrected_script = chain.run({"script": script})

		# Try to load the corrected script
		try:
			local_vars = {"graphviz": graphviz}
			exec(corrected_script, {}, local_vars)
			dot = local_vars["dot"]
			st.success("Graphviz script corrected successfully!")
			return dot
		except Exception as corrected_error:
			st.error(f"Failed to correct the Graphviz script: {corrected_error}")
			return None

# Function to generate SVG data from `dot`
def generate_svg_data(dot):
	# Render the graph as an SVG in memory
	svg_data = dot.pipe(format="svg").decode("utf-8")
	return svg_data

if "response" in st.session_state:
	import graphviz
	import uuid

	# Define the Graphviz script as a Python string
	graph_script = st.session_state["response"]

	# Load and display the graph using the script in the variable
	dot = load_graph_from_script(graph_script, openai.api_key)
	st.graphviz_chart(dot)

	# Button to open the graph as SVG in a new tab
	if st.button("Open SVG in New Tab", key="open_svg_button"):
		# Generate SVG data from the graph
		svg_data = generate_svg_data(dot)

		# Generate a unique identifier for each execution
		unique_id = str(uuid.uuid4())

		# JavaScript to open a new tab and write the SVG directly to the HTML
		js_code = f"""
		<script>
			var svgData = `{svg_data}`;  // Insert SVG content as a string
			var newTab = window.open("about:blank", "_blank");
			newTab.document.write('<html><head><title>SVG Image</title></head><body>' + svgData + '</body></html>');
			newTab.document.close();
		</script>
		"""
		# Display the JavaScript in Streamlit to execute it
		st.components.v1.html(js_code + f"<!-- {unique_id} -->", height=0, width=0)

#st.divider()
#st.write('A project by [Francesco Carlucci](https://francescocarlucci.com) - \
#Need AI training / consulting? [Get in touch](mailto:info@francescocarlucci.com)')